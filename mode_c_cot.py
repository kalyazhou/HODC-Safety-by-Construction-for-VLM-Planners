"""
Mode C with Chain-of-Thought + High-Order Driving Commands
结合Mode B的多步推理 + 高阶驾驶指令的语义可解释性

Architecture:
    Stage 1-3: Scene Understanding (复用Mode B的CoT)
        - SceneDescription: 场景理解
        - DescribeObjects: 物体检测
        - DescribeOrUpdateIntent: 意图推理
    
    Stage 4: Semantic Command Generation (新增语义层)
        - 基于前3步生成高阶驾驶指令（HODC）
    
    Stage 5: Trajectory Generation (精确轨迹)
        - 基于HODC + Context生成[v,k]序列
"""

import json
import numpy as np
from typing import List, Dict, Tuple


def generate_semantic_command_from_cot(
    vlm_inference_fn,
    scene_description: str,
    object_description: str,
    intent_description: str,
    ego_state: Dict,
    history_vk: List[List[float]],
    args
) -> Dict:
    """
    Stage 4: 基于前3步的CoT理解，生成高阶驾驶指令
    
    Args:
        scene_description: Stage 1 场景描述
        object_description: Stage 2 物体信息
        intent_description: Stage 3 意图推理
        ego_state: 自车状态 {speed, curvature}
        history_vk: 历史轨迹
        
    Returns:
        {
            'maneuver': str,
            'curvature_hint': str,
            'speed_target': float,
            'reasoning': str
        }
    """
    curr_speed = ego_state.get('speed', 0.0)
    curr_k = ego_state.get('curvature', 0.0)
    
    sys_message = (
        "You are an expert trajectory planner. Based on multi-step scene analysis, "
        "generate an ENHANCED High-Order Driving Command (HODC++).\n\n"
        "INPUT:\n"
        "- Scene understanding (traffic, lanes, signals)\n"
        "- Object detection (vehicles, pedestrians, risks)\n"
        "- Intent reasoning (goals, maneuvers)\n"
        "- Current ego state\n\n"
        "OUTPUT (strict JSON):\n"
        "{\n"
        '  "maneuver": "straight" | "left_turn" | "right_turn" | "lane_change_left" | "lane_change_right",\n'
        '  "curvature_profile": {\n'
        '    "hint": "straight" | "gentle" | "moderate" | "sharp",\n'
        '    "peak_value": 0.03,  // target peak curvature in 1/m\n'
        '    "shape": "constant" | "bell" | "ramp_up" | "ramp_down"\n'
        '  },\n'
        '  "speed_profile": {\n'
        '    "min_speed": 3.0,  // MINIMUM to maintain progress (m/s) - aim for 2.5-3.5 m/s\n'
        '    "target_speed": 5.0,  // desired cruising speed (m/s) - aim for 4-6 m/s\n'
        '    "max_speed": 7.0,  // speed limit (m/s) - reasonable urban speed\n'
        '    "brake_zones": []  // optional: distances where braking needed, e.g., [10, 15]\n'
        '  },\n'
        '  "safety_constraints": {\n'
        '    "critical_risks": ["pedestrian_crossing", "construction"],  // key risks from Objects\n'
        '    "avoidance_direction": "left" | "right" | "none",\n'
        '    "min_clearance": 2.0  // minimum safe distance to obstacles (m)\n'
        '  },\n'
        '  "reasoning": "Concise explanation linking scene/objects/intent (max 40 words)"\n'
        "}\n\n"
        "HODC++ GENERATION GUIDELINES:\n"
        "1. **Maneuver**: Extract from Intent (e.g., 'turn left' → 'left_turn')\n"
        "2. **Curvature Profile**:\n"
        "   - hint: 'straight'(<0.01), 'gentle'(0.01-0.025), 'moderate'(0.025-0.04), 'sharp'(>0.04)\n"
        "   - peak_value: Infer from Scene lane geometry + current curvature\n"
        "   - shape: 'bell' for turns, 'constant' for straight, 'ramp_up' for lane changes\n"
        "3. **Speed Profile** (Conservative & Safe):\n"
        "   - min_speed: Aim for 2.5-3.5 m/s to maintain progress\n"
        "   - target_speed: Aim for 4-6 m/s for normal driving (safe urban speed)\n"
        "   - max_speed: Use 6-7 m/s for straight roads, lower for turns\n"
        "   - IMPORTANT: Prioritize SAFETY over speed\n"
        "   - IMPORTANT: Reduce speed for pedestrian crossings and high-risk areas\n"
        "   - brake_zones: Mark zones where immediate obstacles within 5-10m\n"
        "4. **Safety Constraints**:\n"
        "   - critical_risks: Extract HIGH/MEDIUM risks from Objects description\n"
        "   - avoidance_direction: Based on risk locations\n"
        "   - min_clearance: 1.5m for static, 2.0m+ for dynamic obstacles\n"
        "5. **Be EXPLICIT but NOT OVERLY CAUTIOUS**: Translate CoT to confident, progress-maintaining commands\n"
    )
    
    user_prompt = (
        f"=== SCENE UNDERSTANDING ===\n{scene_description}\n\n"
        f"=== OBJECT DETECTION ===\n{object_description}\n\n"
        f"=== INTENT REASONING ===\n{intent_description}\n\n"
        f"=== EGO STATE ===\n"
        f"  Speed: {curr_speed:.2f} m/s\n"
        f"  Curvature: {curr_k:.3f} (1/m)\n\n"
        f"Generate semantic driving command:"
    )
    
    try:
        response = vlm_inference_fn(
            text=user_prompt,
            images=None,  # No images, rely on CoT text
            args=args,
            sys_message=sys_message,
            model_type="gpt-4o-mini",
            resp_tokens=250,  # Increased for HODC++
            force_json=True
        )
        
        result = _parse_json_response(response)
        
        # Parse HODC++ structure
        maneuver = result.get('maneuver', 'straight')
        curv_profile = result.get('curvature_profile', {})
        speed_profile = result.get('speed_profile', {})
        safety_constraints = result.get('safety_constraints', {})
        reasoning = result.get('reasoning', '')
        
        # Extract key fields
        curvature_hint = curv_profile.get('hint', 'straight')
        peak_curvature = float(curv_profile.get('peak_value', 0.02))
        curv_shape = curv_profile.get('shape', 'bell')
        
        # 实验1：保守速度参数（安全导向）
        min_speed = float(speed_profile.get('min_speed', 3.0))
        target_speed = float(speed_profile.get('target_speed', 5.0))
        max_speed = float(speed_profile.get('max_speed', 7.0))
        
        # 确保逻辑一致性
        if target_speed < min_speed:
            target_speed = min_speed + 1.0
        if max_speed < target_speed:
            max_speed = target_speed + 1.0
        
        brake_zones = speed_profile.get('brake_zones', [])
        
        critical_risks = safety_constraints.get('critical_risks', [])
        avoidance_dir = safety_constraints.get('avoidance_direction', 'none')
        min_clearance = float(safety_constraints.get('min_clearance', 1.5))
        
        print(f"[HODC++] Maneuver: {maneuver}, Curvature: {curvature_hint}(peak={peak_curvature:.3f}, {curv_shape})")
        print(f"[HODC++] Speed: min={min_speed:.1f}, target={target_speed:.1f}, max={max_speed:.1f}m/s")
        print(f"[HODC++] Safety: risks={critical_risks}, avoid={avoidance_dir}, clearance={min_clearance:.1f}m")
        print(f"[HODC++] Reasoning: {reasoning}")
        
        return {
            'maneuver': maneuver,
            'curvature_hint': curvature_hint,
            'peak_curvature': peak_curvature,
            'curvature_shape': curv_shape,
            'min_speed': min_speed,
            'target_speed': target_speed,
            'max_speed': max_speed,
            'brake_zones': brake_zones,
            'critical_risks': critical_risks,
            'avoidance_direction': avoidance_dir,
            'min_clearance': min_clearance,
            'reasoning': reasoning,
            # Legacy compatibility
            'speed_target': target_speed
        }
        
    except Exception as e:
        print(f"[HODC++] Parse error: {e}, using fallback")
        # Fallback: 基于意图描述和当前状态推理
        maneuver = 'straight'
        if 'left' in intent_description.lower():
            maneuver = 'left_turn'
        elif 'right' in intent_description.lower():
            maneuver = 'right_turn'
        
        # 基于当前曲率推理
        if abs(curr_k) > 0.04:
            curvature_hint = 'sharp'
            peak_curvature = abs(curr_k) * 1.1
        elif abs(curr_k) > 0.025:
            curvature_hint = 'moderate'
            peak_curvature = abs(curr_k) * 1.1
        elif abs(curr_k) > 0.01:
            curvature_hint = 'gentle'
            peak_curvature = max(abs(curr_k) * 1.1, 0.02)
        else:
            curvature_hint = 'straight'
            peak_curvature = 0.0
        
        # 速度推理（实验1：保守安全导向）
        if curvature_hint == 'straight':
            target_speed = min(curr_speed * 1.1, 5.0)  # 保守的直线速度
            min_speed = max(curr_speed * 0.9, 3.0)  # 保守最小速度
        else:
            target_speed = max(curr_speed * 0.9, 4.0)  # 转弯时保守
            min_speed = max(curr_speed * 0.7, 2.5)  # 转弯最小速度保守
        max_speed = min(target_speed * 1.4, 7.0)  # 保守最大速度上限
        
        # 安全约束（保守fallback）
        critical_risks = ['unknown_risk']
        avoidance_dir = 'none'
        min_clearance = 2.0
        
        return {
            'maneuver': maneuver,
            'curvature_hint': curvature_hint,
            'peak_curvature': peak_curvature,
            'curvature_shape': 'bell' if curvature_hint != 'straight' else 'constant',
            'min_speed': min_speed,
            'target_speed': target_speed,
            'max_speed': max_speed,
            'brake_zones': [],
            'critical_risks': critical_risks,
            'avoidance_direction': avoidance_dir,
            'min_clearance': min_clearance,
            'reasoning': f'Fallback: {maneuver} based on intent+curvature',
            # Legacy compatibility
            'speed_target': target_speed
        }


def generate_trajectory_from_hodc_and_context(
    vlm_inference_fn,
    semantic_cmd: Dict,
    scene_description: str,
    object_description: str,
    ego_state: Dict,
    history_vk: List[List[float]],
    args
) -> List[List[float]]:
    """
    Stage 5: 基于HODC + 场景上下文生成精确[v,k]轨迹
    
    Args:
        semantic_cmd: Stage 4的语义指令
        scene_description: Stage 1的场景理解
        object_description: Stage 2的物体信息
        ego_state: 自车状态
        history_vk: 历史轨迹
        
    Returns:
        [[v1,k1], ..., [v10,k10]]
    """
    # Extract HODC++ fields
    maneuver = semantic_cmd['maneuver']
    curvature_hint = semantic_cmd['curvature_hint']
    peak_curvature = semantic_cmd.get('peak_curvature', 0.02)
    curv_shape = semantic_cmd.get('curvature_shape', 'bell')
    min_speed = semantic_cmd.get('min_speed', 2.5)
    target_speed = semantic_cmd.get('target_speed', 4.0)
    max_speed = semantic_cmd.get('max_speed', 6.0)
    brake_zones = semantic_cmd.get('brake_zones', [])
    critical_risks = semantic_cmd.get('critical_risks', [])
    avoidance_dir = semantic_cmd.get('avoidance_direction', 'none')
    min_clearance = semantic_cmd.get('min_clearance', 1.5)
    reasoning = semantic_cmd.get('reasoning', '')
    
    curr_speed = ego_state.get('speed', 0.0)
    curr_k = ego_state.get('curvature', 0.0)
    
    # Determine direction
    if 'left' in maneuver:
        k_sign = 1
    elif 'right' in maneuver:
        k_sign = -1
    else:
        k_sign = 0
    
    # Convert peak_curvature to ×100 format
    k_peak_scaled = peak_curvature * 100 * k_sign
    
    # Build system message with HODC++ constraints
    sys_message = (
        "You are a precision trajectory generator. Convert ENHANCED High-Order Driving Commands (HODC++) "
        "into precise [v,k] pairs that satisfy all geometric, speed, and safety constraints.\n\n"
        "=== HODC++ INPUT ===\n"
        f"Maneuver: {maneuver}\n"
        f"Curvature Profile: {curvature_hint} (peak={peak_curvature:.3f} 1/m, shape={curv_shape})\n"
        f"Speed Constraints: min={min_speed:.1f}, target={target_speed:.1f}, max={max_speed:.1f} m/s\n"
        f"Safety: risks={critical_risks}, avoidance={avoidance_dir}, min_clearance={min_clearance:.1f}m\n"
        f"Brake Zones: {brake_zones if brake_zones else 'none'}\n"
        f"Reasoning: {reasoning}\n\n"
        "OUTPUT (strict JSON):\n"
        "{\n"
        '  "pairs": [[v1, k1], [v2, k2], ..., [v10, k10]]  // k in ×100 format\n'
        "}\n\n"
        "=== TRAJECTORY GENERATION CONSTRAINTS ===\n"
    )
    
    if k_sign != 0:
        # Curvature profile based on shape
        if curv_shape == 'bell':
            k_example = f"[0, {k_peak_scaled*0.3:.0f}, {k_peak_scaled*0.6:.0f}, {k_peak_scaled*0.9:.0f}, {k_peak_scaled:.0f}, {k_peak_scaled*0.9:.0f}, {k_peak_scaled*0.6:.0f}, {k_peak_scaled*0.3:.0f}, 0, 0]"
        elif curv_shape == 'ramp_up':
            k_example = f"[0, {k_peak_scaled*0.2:.0f}, {k_peak_scaled*0.4:.0f}, {k_peak_scaled*0.6:.0f}, {k_peak_scaled*0.8:.0f}, {k_peak_scaled:.0f}, {k_peak_scaled:.0f}, {k_peak_scaled:.0f}, {k_peak_scaled:.0f}, {k_peak_scaled:.0f}]"
        elif curv_shape == 'ramp_down':
            k_example = f"[{k_peak_scaled:.0f}, {k_peak_scaled:.0f}, {k_peak_scaled:.0f}, {k_peak_scaled*0.8:.0f}, {k_peak_scaled*0.6:.0f}, {k_peak_scaled*0.4:.0f}, {k_peak_scaled*0.2:.0f}, 0, 0, 0]"
        else:  # constant
            k_example = f"[{k_peak_scaled:.0f}] * 10"
        
        sys_message += (
            f"1. **Curvature Profile** ({curv_shape}):\n"
            f"   - Peak value: {k_peak_scaled:.0f} (×100 format, = {peak_curvature:.3f} 1/m)\n"
            f"   - Example sequence: {k_example}\n"
            f"   - Smoothness: Δk ≤ 200 per step (0.5s)\n"
            f"2. **Speed Profile**:\n"
            f"   - Start from current: {curr_speed:.1f} m/s\n"
            f"   - Target: {target_speed:.1f} m/s\n"
            f"   - HARD LIMITS: [{min_speed:.1f}, {max_speed:.1f}] m/s\n"
            f"   - Brake zones: {brake_zones if brake_zones else 'none'}\n"
            f"   - Smoothness: Δv ≤ 1.0 m/s per 0.5s\n"
            f"3. **Safety Constraints**:\n"
            f"   - Critical risks: {', '.join(critical_risks) if critical_risks else 'none'}\n"
            f"   - If risks present: reduce speed by 10-20%, increase caution\n"
            f"   - Avoidance direction: {avoidance_dir}\n"
            f"   - Maintain clearance: ≥{min_clearance:.1f}m\n"
            f"4. **CRITICAL**: \n"
            f"   - DO NOT output zero curvature if peak_curvature > 0.01\n"
            f"   - DO NOT exceed speed limits even if target is higher\n"
            f"   - Prioritize safety over progress\n"
        )
    else:
        sys_message += (
            f"1. **Curvature**: All k=0 (straight driving)\n"
            f"2. **Speed Profile**:\n"
            f"   - Start from current: {curr_speed:.1f} m/s\n"
            f"   - Target: {target_speed:.1f} m/s\n"
            f"   - HARD LIMITS: [{min_speed:.1f}, {max_speed:.1f}] m/s\n"
            f"   - Gradual S-curve acceleration\n"
            f"   - Smoothness: Δv ≤ 1.0 m/s per 0.5s\n"
            f"3. **Safety Constraints**:\n"
            f"   - Critical risks: {', '.join(critical_risks) if critical_risks else 'none'}\n"
            f"   - If risks present: reduce speed, be ready to brake\n"
            f"   - Min clearance: {min_clearance:.1f}m\n"
        )
    
    # Extract critical risk details from object description
    risk_summary = "No specific risks" if not critical_risks else f"CRITICAL: {', '.join(critical_risks)}"
    
    user_prompt = (
        f"=== EGO STATE ===\n"
        f"  Current: v={curr_speed:.2f}m/s, k={curr_k:.3f} (1/m)\n"
        f"  History: {history_vk[-3:] if len(history_vk)>=3 else history_vk}\n\n"
        f"=== CRITICAL RISKS (from CoT analysis) ===\n"
        f"{risk_summary}\n"
        f"Avoidance: {avoidance_dir}, Min clearance: {min_clearance:.1f}m\n\n"
        f"=== TASK ===\n"
        f"Generate 10 [v,k] pairs that:\n"
        f"1. Follow the HODC++ curvature profile ({curv_shape}, peak={peak_curvature:.3f})\n"
        f"2. Respect speed constraints (min={min_speed:.1f}, target={target_speed:.1f}, max={max_speed:.1f})\n"
        f"3. AVOID all critical risks by adjusting speed/trajectory\n"
        f"4. Maintain smoothness (Δv≤1.0, Δk≤200 per 0.5s)\n\n"
        f"Generate the trajectory now:"
    )
    
    try:
        response = vlm_inference_fn(
            text=user_prompt,
            images=None,
            args=args,
            sys_message=sys_message,
            model_type="gpt-4o-mini",
            resp_tokens=200,
            force_json=True
        )
        
        result = _parse_json_response(response)
        pairs = result.get('pairs', [])
        
        if len(pairs) < 10:
            print(f"[Traj] Warning: Only got {len(pairs)} pairs, padding...")
            while len(pairs) < 10:
                pairs.append([speed_target * 0.8, 0])
        
        pairs = pairs[:10]
        
        # Scale curvature from ×100 to 1/m
        for i, (v, k) in enumerate(pairs):
            pairs[i] = [float(v), float(k) / 100.0]
        
        # CRITICAL: Validate VLM output against HODC++ constraints
        k_values = [abs(p[1]) for p in pairs]
        k_max_actual = max(k_values)
        k_target = abs(peak_curvature)
        
        # First check: HODC++ internal consistency (maneuver vs peak_curvature)
        # 实验3：适度的一致性检查（介于实验1和2之间）
        if k_target > 0.02 and maneuver == 'straight':  # 中等阈值（实验1:0.015, 实验2:0.025）
            print(f"[Traj] HODC++ inconsistency detected: maneuver='straight' but peak={k_target:.3f}")
            print(f"[Traj] Correcting label (but still using VLM output)")
            # Auto-correct maneuver based on current ego curvature
            curr_k = ego_state.get('curvature', 0.0)
            if curr_k > 0.005:
                semantic_cmd['maneuver'] = 'left_turn'
                k_sign = 1
            elif curr_k < -0.005:
                semantic_cmd['maneuver'] = 'right_turn'
                k_sign = -1
            else:
                # Default to left turn if no ego curvature hint
                semantic_cmd['maneuver'] = 'left_turn'
                k_sign = 1
            print(f"[Traj] Label corrected to: '{semantic_cmd['maneuver']}' (not forcing fallback)")
        
        # Second check: VLM output vs HODC++ target
        # 实验4：保持实验3的验证阈值
        should_validate = (k_target > 0.02)
        
        if should_validate:
            deviation_ratio = abs(k_max_actual - k_target) / max(k_target, 0.001)
            # For turn maneuvers, check if VLM violated constraints severely
            if k_sign != 0 and (deviation_ratio > 4.0 or k_max_actual < 0.002):
                print(f"[Traj] VLM violated curvature constraint: target={k_target:.3f}, actual={k_max_actual:.3f} (deviation={deviation_ratio:.1%})")
                print(f"[Traj] Falling back to HODC++ template to enforce constraints")
                return _generate_template_fallback(semantic_cmd, ego_state)
        
        # 实验1：No Post-Processing，完全依赖VLM输出和strict fallback
        print(f"[Traj] Generated {len(pairs)} pairs from HODC+Context (no post-processing, k_peak={max(abs(p[1]) for p in pairs):.3f})")
        return pairs
        
    except Exception as e:
        print(f"[Traj] Parse error: {e}, using template fallback")
        return _generate_template_fallback(semantic_cmd, ego_state)


def _parse_json_response(response: str) -> Dict:
    """解析VLM的JSON响应"""
    try:
        # Try direct JSON parse
        return json.loads(response)
    except:
        # Extract JSON from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*"[^"]*"[^{}]*:[^{}]*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        
        raise ValueError("No valid JSON found in response")


def _generate_template_fallback(semantic_cmd: Dict, ego_state: Dict) -> List[List[float]]:
    """Template-based fallback trajectory generation using HODC++"""
    maneuver = semantic_cmd['maneuver']
    curvature_hint = semantic_cmd['curvature_hint']
    peak_curvature = semantic_cmd.get('peak_curvature', 0.02)
    curv_shape = semantic_cmd.get('curvature_shape', 'bell')
    min_speed = semantic_cmd.get('min_speed', 2.5)  # 实验1：保守速度（与Stage 4保持一致）
    target_speed = semantic_cmd.get('target_speed', 4.0)  # 实验1：保守速度（与Stage 4保持一致）
    max_speed = semantic_cmd.get('max_speed', 6.0)  # 实验1：保守速度（与Stage 4保持一致）
    critical_risks = semantic_cmd.get('critical_risks', [])
    
    curr_speed = ego_state.get('speed', 3.0)
    
    # Generate curvature profile based on shape
    if curv_shape == 'bell':
        # Bell-shaped: smooth peak in middle
        bell = np.sin(np.linspace(0, np.pi, 10))
        k_template = (peak_curvature * bell).tolist()
    elif curv_shape == 'ramp_up':
        # Ramp up: gradually increase to peak
        k_template = (peak_curvature * np.linspace(0, 1, 10)).tolist()
    elif curv_shape == 'ramp_down':
        # Ramp down: gradually decrease from peak
        k_template = (peak_curvature * np.linspace(1, 0, 10)).tolist()
    elif curv_shape == 'constant':
        # Constant curvature
        k_template = [peak_curvature] * 10
    else:
        # Default: straight
        k_template = [0] * 10
    
    # Apply direction
    if 'right' in maneuver:
        k_template = [-k for k in k_template]
    
    # 实验1：简单的S-curve速度策略（保守、安全导向）
    # 生成从当前速度到目标速度的平滑S-curve
    v_profile = []
    for i in range(10):
        t = i / 9.0
        # Sigmoid插值：前段保持当前速度，后段趋向目标速度
        progress = 1 / (1 + np.exp(-10 * (t - 0.5)))
        v = curr_speed + (target_speed - curr_speed) * progress
        v = np.clip(v, min_speed, max_speed)
        # 如果有风险，额外降速（保守策略）
        if critical_risks:
            v = max(v * 0.8, min_speed)  # 降速20%（比实验5更保守）
        v_profile.append(float(v))
    
    print(f"[Template] Generated S-curve: {curr_speed:.1f} → {target_speed:.1f} m/s (safe, conservative)")
    
    return [[v, k] for v, k in zip(v_profile, k_template)]


def generate_mode_c_cot_prediction(
    vlm_inference_fn,
    images: List,
    history_vk: List[List[float]],
    ego_state: Dict,
    scene_info: Dict,
    scene_description: str,
    object_description: str,
    intent_description: str,
    args
) -> Dict:
    """
    完整的Mode C (CoT+HODC) 预测流程
    
    Args:
        vlm_inference_fn: VLM推理函数
        images: 图像列表（可选，主要用于前3步）
        history_vk: 历史轨迹
        ego_state: 自车状态
        scene_info: 场景信息
        scene_description: Stage 1输出
        object_description: Stage 2输出
        intent_description: Stage 3输出
        args: 参数
        
    Returns:
        {
            'pairs': [[v,k], ...],
            'semantic_command': {...},
            'reasoning': str
        }
    """
    # Stage 4: Generate semantic command from CoT
    semantic_cmd = generate_semantic_command_from_cot(
        vlm_inference_fn,
        scene_description,
        object_description,
        intent_description,
        ego_state,
        history_vk,
        args
    )
    
    # Stage 5: Generate trajectory from HODC + Context
    pairs = generate_trajectory_from_hodc_and_context(
        vlm_inference_fn,
        semantic_cmd,
        scene_description,
        object_description,
        ego_state,
        history_vk,
        args
    )
    
    return {
        'pairs': pairs,
        'semantic_command': semantic_cmd,
        'reasoning': semantic_cmd.get('reasoning', ''),
        'constraints': {}  # For compatibility
    }

