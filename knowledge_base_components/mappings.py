import re
from typing import Dict, List


def build_keyword_mapping() -> Dict[str, List[str]]:
    """
    构建关键词到对象类型的映射库。
    这是一个语义知识库，将常见用户描述映射到标准 EnergyPlus 对象。
    """
    return {
        # 照明相关
        "照明": ["Lights"],
        "照明功率": ["Lights"],
        "照明密度": ["Lights"],
        "灯光": ["Lights"],
        "lighting": ["Lights"],
        "light": ["Lights"],

        # 墙体材料相关
        "墙体": ["Material", "Material:NoMass", "Construction"],
        "外墙": ["Material", "Material:NoMass", "Construction"],
        "墙": ["Material", "Material:NoMass", "Construction"],
        "材料": ["Material", "Material:NoMass"],
        "导热系数": ["Material", "Material:NoMass"],
        "热阻": ["Material:NoMass"],
        "保温": ["Material", "Material:NoMass"],
        "wall": ["Material", "Material:NoMass", "Construction"],
        "material": ["Material", "Material:NoMass"],

        # 窗户相关
        "窗": ["WindowMaterial:SimpleGlazingSystem", "WindowMaterial:Glazing", "FenestrationSurface:Detailed"],
        "窗户": ["WindowMaterial:SimpleGlazingSystem", "WindowMaterial:Glazing", "FenestrationSurface:Detailed"],
        "玻璃": ["WindowMaterial:Glazing", "WindowMaterial:SimpleGlazingSystem"],
        "遮阳": ["WindowMaterial:Glazing", "Shading:Building:Detailed"],
        "遮阳系数": ["WindowMaterial:SimpleGlazingSystem", "WindowMaterial:Glazing"],
        "太阳辐射": ["WindowMaterial:SimpleGlazingSystem", "WindowMaterial:Glazing"],
        "SHGC": ["WindowMaterial:SimpleGlazingSystem"],
        "window": ["WindowMaterial:SimpleGlazingSystem", "WindowMaterial:Glazing"],
        "glazing": ["WindowMaterial:Glazing", "WindowMaterial:SimpleGlazingSystem"],

        # 渗透相关
        "渗透": ["ZoneInfiltration:DesignFlowRate"],
        "空气渗透": ["ZoneInfiltration:DesignFlowRate"],
        "渗透率": ["ZoneInfiltration:DesignFlowRate"],
        "infiltration": ["ZoneInfiltration:DesignFlowRate"],

        # 设备相关
        "设备": ["OtherEquipment", "ElectricEquipment", "GasEquipment"],
        "电气设备": ["OtherEquipment", "ElectricEquipment"],
        "设备功率": ["OtherEquipment", "ElectricEquipment", "GasEquipment"],
        "设备密度": ["OtherEquipment", "ElectricEquipment", "GasEquipment"],
        "equipment": ["OtherEquipment", "ElectricEquipment", "GasEquipment"],

        # 人员相关
        "人员": ["People"],
        "人员密度": ["People"],
        "people": ["People"],

        # HVAC相关
        "空调": ["ZoneHVAC:IdealLoadsAirSystem", "AirLoopHVAC"],
        "暖通": ["ZoneHVAC:IdealLoadsAirSystem", "AirLoopHVAC"],
        "供暖": ["Heating:DesignDay", "ZoneHVAC:IdealLoadsAirSystem"],
        "制冷": ["Cooling:DesignDay", "ZoneHVAC:IdealLoadsAirSystem"],
        "hvac": ["ZoneHVAC:IdealLoadsAirSystem", "AirLoopHVAC"],
        "vrf": [
            "AirConditioner:VariableRefrigerantFlow",
            "AirConditioner:VariableRefrigerantFlow:FluidTemperatureControl",
            "ZoneHVAC:TerminalUnit:VariableRefrigerantFlow",
            "ZoneHVAC:IdealLoadsAirSystem",
        ],
        "多联机": [
            "AirConditioner:VariableRefrigerantFlow",
            "AirConditioner:VariableRefrigerantFlow:FluidTemperatureControl",
            "ZoneHVAC:TerminalUnit:VariableRefrigerantFlow",
            "ZoneHVAC:IdealLoadsAirSystem",
        ],
        "变冷媒": [
            "AirConditioner:VariableRefrigerantFlow",
            "AirConditioner:VariableRefrigerantFlow:FluidTemperatureControl",
            "ZoneHVAC:TerminalUnit:VariableRefrigerantFlow",
            "ZoneHVAC:IdealLoadsAirSystem",
        ],
        "理想空气": ["ZoneHVAC:IdealLoadsAirSystem"],

        # 日光与空气分配
        "采光": ["Daylighting:Controls"],
        "日光": ["Daylighting:Controls"],
        "daylighting": ["Daylighting:Controls"],
        "照度控制": ["Daylighting:Controls"],
        "空气分配": ["DesignSpecification:ZoneAirDistribution"],
        "送风效率": ["DesignSpecification:ZoneAirDistribution"],
        "通风效率": ["DesignSpecification:ZoneAirDistribution"],

        # 温控设定点相关
        "温控": ["ThermostatSetpoint:DualSetpoint"],
        "设定点": ["ThermostatSetpoint:DualSetpoint"],
        "温度设定": ["ThermostatSetpoint:DualSetpoint"],
        "供暖温度": ["ThermostatSetpoint:DualSetpoint", "Sizing:Zone"],
        "制冷温度": ["ThermostatSetpoint:DualSetpoint", "Sizing:Zone"],
        "thermostat": ["ThermostatSetpoint:DualSetpoint"],
        "setpoint": ["ThermostatSetpoint:DualSetpoint"],

        # 新风量相关
        "新风": ["DesignSpecification:OutdoorAir"],
        "新风量": ["DesignSpecification:OutdoorAir"],
        "室外空气": ["DesignSpecification:OutdoorAir"],
        "通风": ["DesignSpecification:OutdoorAir"],
        "ventilation": ["DesignSpecification:OutdoorAir"],
        "outdoor air": ["DesignSpecification:OutdoorAir"],

        # 热回收相关
        "热回收": ["ZoneHVAC:IdealLoadsAirSystem"],
        "能量回收": ["ZoneHVAC:IdealLoadsAirSystem"],
        "全热回收": ["ZoneHVAC:IdealLoadsAirSystem"],
        "热回收效率": ["ZoneHVAC:IdealLoadsAirSystem"],
        "显热回收": ["ZoneHVAC:IdealLoadsAirSystem"],
        "潜热回收": ["ZoneHVAC:IdealLoadsAirSystem"],
        "显热效率": ["ZoneHVAC:IdealLoadsAirSystem"],
        "潜热效率": ["ZoneHVAC:IdealLoadsAirSystem"],
        "热交换": ["ZoneHVAC:IdealLoadsAirSystem"],
        "回收效率": ["ZoneHVAC:IdealLoadsAirSystem"],
        "HRV": ["ZoneHVAC:IdealLoadsAirSystem"],
        "ERV": ["ZoneHVAC:IdealLoadsAirSystem"],
        "heat recovery": ["ZoneHVAC:IdealLoadsAirSystem"],

        # 区域设计参数
        "设计参数": ["Sizing:Zone"],
        "供风温度": ["Sizing:Zone", "ZoneHVAC:IdealLoadsAirSystem"],
        "设计供风温度": ["Sizing:Zone"],
        "供冷温度": ["Sizing:Zone"],
        "供暖供冷温度": ["Sizing:Zone"],
        "sizing": ["Sizing:Zone"],

        # IdealLoads 阈值相关
        "阈值": ["ZoneHVAC:IdealLoadsAirSystem"],
        "最大温度": ["ZoneHVAC:IdealLoadsAirSystem"],
        "最小温度": ["ZoneHVAC:IdealLoadsAirSystem"],
        "上限温度": ["ZoneHVAC:IdealLoadsAirSystem"],
        "下限温度": ["ZoneHVAC:IdealLoadsAirSystem"],
        "温度上限": ["ZoneHVAC:IdealLoadsAirSystem"],
        "温度下限": ["ZoneHVAC:IdealLoadsAirSystem"],

        # 围护结构
        "围护结构": ["Construction"],
        "构造": ["Construction"],
        "construction": ["Construction"],
    }


def build_field_keyword_mapping() -> Dict[str, Dict[str, List[str]]]:
    """
    构建粗粒度的关键词到字段映射。
    格式: {object_type: {keyword: [field_names]}}
    """
    return {
        "Lights": {
            "功率": ["Watts_per_Zone_Floor_Area", "Watts_per_Floor_Area", "Watts_per_Person", "Lighting_Level"],
            "密度": ["Watts_per_Zone_Floor_Area", "Watts_per_Floor_Area", "Watts_per_Person"],
            "瓦特": ["Watts_per_Zone_Floor_Area", "Watts_per_Floor_Area", "Watts_per_Person", "Lighting_Level"],
            "watts": ["Watts_per_Zone_Floor_Area", "Watts_per_Floor_Area", "Watts_per_Person", "Lighting_Level"],
            "power": ["Watts_per_Zone_Floor_Area", "Watts_per_Floor_Area", "Watts_per_Person", "Lighting_Level"],
        },
        "OtherEquipment": {
            "功率": ["Power_per_Zone_Floor_Area", "Power_per_Person", "Design_Level"],
            "密度": ["Power_per_Zone_Floor_Area", "Power_per_Person"],
            "瓦特": ["Power_per_Zone_Floor_Area", "Power_per_Person", "Design_Level"],
            "watts": ["Power_per_Zone_Floor_Area", "Power_per_Person", "Design_Level"],
            "power": ["Power_per_Zone_Floor_Area", "Power_per_Person", "Design_Level"],
            "equipment": ["Power_per_Zone_Floor_Area", "Power_per_Person", "Design_Level"],
        },
        "Material": {
            "导热": ["Conductivity"],
            "导热系数": ["Conductivity"],
            "热导率": ["Conductivity"],
            "conductivity": ["Conductivity"],
            "厚度": ["Thickness"],
            "thickness": ["Thickness"],
            "密度": ["Density"],
            "density": ["Density"],
        },
        "Material:NoMass": {
            "热阻": ["Thermal_Resistance"],
            "resistance": ["Thermal_Resistance"],
        },
        "WindowMaterial:SimpleGlazingSystem": {
            "U值": ["UFactor"],
            "传热系数": ["UFactor"],
            "ufactor": ["UFactor"],
            "shgc": ["Solar_Heat_Gain_Coefficient"],
            "遮阳系数": ["Solar_Heat_Gain_Coefficient"],
            "太阳热得": ["Solar_Heat_Gain_Coefficient"],
            "可见光": ["Visible_Transmittance"],
            "透光": ["Visible_Transmittance"],
        },
        "WindowMaterial:Glazing": {
            "光学": ["Solar_Transmittance_at_Normal_Incidence", "Front_Side_Solar_Reflectance_at_Normal_Incidence", "Back_Side_Solar_Reflectance_at_Normal_Incidence"],
            "厚度": ["Thickness"],
        },
        "ZoneInfiltration:DesignFlowRate": {
            "渗透": ["Air_Changes_per_Hour", "Flow_per_Exterior_Surface_Area", "Flow_Rate_per_Exterior_Surface_Area", "Flow_per_Zone_Floor_Area", "Flow_Rate_per_Floor_Area"],
            "渗透率": ["Air_Changes_per_Hour", "Flow_per_Exterior_Surface_Area", "Flow_Rate_per_Exterior_Surface_Area", "Flow_per_Zone_Floor_Area", "Flow_Rate_per_Floor_Area"],
            "换气": ["Air_Changes_per_Hour"],
            "infiltration": ["Air_Changes_per_Hour", "Flow_per_Exterior_Surface_Area", "Flow_Rate_per_Exterior_Surface_Area", "Flow_per_Zone_Floor_Area", "Flow_Rate_per_Floor_Area"],
        },
        "ElectricEquipment": {
            "功率": ["Watts_per_Floor_Area", "Watts_per_Person"],
            "密度": ["Watts_per_Floor_Area", "Watts_per_Person"],
            "equipment": ["Watts_per_Floor_Area", "Watts_per_Person"],
        },
        "People": {
            "人员密度": ["People_per_Zone_Floor_Area", "People_per_Floor_Area"],
            "人均面积": ["Zone_Floor_Area_per_Person", "Floor_Area_per_Person"],
            "density": ["People_per_Zone_Floor_Area", "People_per_Floor_Area"],
        },
        "ThermostatSetpoint:DualSetpoint": {
            "供暖": ["Heating_Setpoint_Temperature_Schedule_Name"],
            "制冷": ["Cooling_Setpoint_Temperature_Schedule_Name"],
            "heating": ["Heating_Setpoint_Temperature_Schedule_Name"],
            "cooling": ["Cooling_Setpoint_Temperature_Schedule_Name"],
        },
        "DesignSpecification:OutdoorAir": {
            "新风": ["Outdoor_Air_Flow_per_Person", "Outdoor_Air_Flow_per_Zone_Floor_Area", "Outdoor_Air_Flow_Air_Changes_per_Hour"],
            "人均新风": ["Outdoor_Air_Flow_per_Person"],
            "换气次数": ["Outdoor_Air_Flow_Air_Changes_per_Hour"],
            "outdoor air": ["Outdoor_Air_Flow_per_Person"],
        },
        "DesignSpecification:ZoneAirDistribution": {
            "送风效率": ["Zone_Air_Distribution_Effectiveness_in_Cooling_Mode", "Zone_Air_Distribution_Effectiveness_in_Heating_Mode"],
            "空气分配": ["Zone_Air_Distribution_Effectiveness_in_Cooling_Mode", "Zone_Air_Distribution_Effectiveness_in_Heating_Mode"],
            "通风效率": ["Zone_Air_Distribution_Effectiveness_in_Cooling_Mode", "Zone_Air_Distribution_Effectiveness_in_Heating_Mode"],
        },
        "Daylighting:Controls": {
            "照度": ["Illuminance_Setpoint_at_Reference_Point_1", "Illuminance_Setpoint_at_Reference_Point_2"],
            "日光": ["Minimum_Input_Power_Fraction_for_Continuous_or_ContinuousOff_Dimming_Control", "Minimum_Light_Output_Fraction_for_Continuous_or_ContinuousOff_Dimming_Control"],
            "调光": ["Minimum_Input_Power_Fraction_for_Continuous_or_ContinuousOff_Dimming_Control", "Minimum_Light_Output_Fraction_for_Continuous_or_ContinuousOff_Dimming_Control"],
        },
        "ZoneHVAC:IdealLoadsAirSystem": {
            "阈值": ["Minimum_Cooling_Supply_Air_Temperature", "Maximum_Heating_Supply_Air_Temperature"],
            "供风阈值": ["Minimum_Cooling_Supply_Air_Temperature", "Maximum_Heating_Supply_Air_Temperature"],
            "最小制冷供风温度": ["Minimum_Cooling_Supply_Air_Temperature"],
            "最大供暖供风温度": ["Maximum_Heating_Supply_Air_Temperature"],
            "热回收": ["Sensible_Heat_Recovery_Effectiveness", "Latent_Heat_Recovery_Effectiveness"],
            "全热回收": ["Sensible_Heat_Recovery_Effectiveness", "Latent_Heat_Recovery_Effectiveness"],
            "热回收效率": ["Sensible_Heat_Recovery_Effectiveness", "Latent_Heat_Recovery_Effectiveness"],
            "显热效率": ["Sensible_Heat_Recovery_Effectiveness"],
            "潜热效率": ["Latent_Heat_Recovery_Effectiveness"],
            "供风含湿量": ["Minimum_Cooling_Supply_Air_Humidity_Ratio", "Maximum_Heating_Supply_Air_Humidity_Ratio"],
            "湿度比": ["Minimum_Cooling_Supply_Air_Humidity_Ratio", "Maximum_Heating_Supply_Air_Humidity_Ratio"],
        },
        "Sizing:Zone": {
            "供风温度": ["Zone_Cooling_Design_Supply_Air_Temperature", "Zone_Heating_Design_Supply_Air_Temperature"],
            "制冷温度": ["Zone_Cooling_Design_Supply_Air_Temperature"],
            "供暖温度": ["Zone_Heating_Design_Supply_Air_Temperature"],
            "供风含湿量": ["Zone_Cooling_Design_Supply_Air_Humidity_Ratio", "Zone_Heating_Design_Supply_Air_Humidity_Ratio"],
            "湿度比": ["Zone_Cooling_Design_Supply_Air_Humidity_Ratio", "Zone_Heating_Design_Supply_Air_Humidity_Ratio"],
            "裕量": ["Zone_Cooling_Sizing_Factor", "Zone_Heating_Sizing_Factor"],
            "安全系数": ["Zone_Cooling_Sizing_Factor", "Zone_Heating_Sizing_Factor"],
        },
    }


def build_field_semantics() -> Dict[str, Dict[str, Dict]]:
    """构建字段语义库。"""
    return {
        "WindowMaterial:Glazing": {
            "Solar_Transmittance_at_Normal_Incidence": {
                "description": "太阳光透射率：表示有多少百分比的太阳光能透射进入室内",
                "unit": "无量纲 (0-1)",
                "range": [0, 1],
                "semantic": "值越高，越多太阳热进入室内",
                "related_concept": ["透射率", "太阳透射", "透光性"],
            },
            "Front_Side_Solar_Reflectance_at_Normal_Incidence": {
                "description": "前面太阳反射率：表示玻璃前面反射的太阳光比例，用于遮挡太阳辐射",
                "unit": "无量纲 (0-1)",
                "range": [0, 1],
                "semantic": "值越高，越多太阳光被反射回去，遮阳效果越好",
                "related_concept": ["遮阳系数", "太阳遮挡", "隔热", "反射率"],
            },
            "Back_Side_Solar_Reflectance_at_Normal_Incidence": {
                "description": "后面太阳反射率：表示玻璃后面反射的太阳光比例",
                "unit": "无量纲 (0-1)",
                "range": [0, 1],
                "semantic": "值越高，越多太阳光被反射，遮阳效果越好",
                "related_concept": ["遮阳系数", "太阳遮挡"],
            },
            "Thickness": {
                "description": "玻璃厚度",
                "unit": "mm",
                "range": [0, 100],
                "semantic": "厚度越大，隔音隔热效果越好，但成本增加",
                "related_concept": ["厚度", "保温"],
            },
        },
        "WindowMaterial:SimpleGlazingSystem": {
            "Solar_Heat_Gain_Coefficient": {
                "description": "太阳热得系数(SHGC)：综合考虑透射和吸收，表示通过窗户进入室内的太阳热比例",
                "unit": "无量纲 (0-1)",
                "range": [0, 1],
                "semantic": "值越低，进入室内的太阳热越少，夏季隔热效果越好",
                "related_concept": ["遮阳系数", "SHGC", "太阳热得系数", "隔热"],
            },
            "UFactor": {
                "description": "传热系数(U值)：表示窗户的保温性能，值越低隔热性能越好",
                "unit": "W/m²K",
                "range": [0.1, 6],
                "semantic": "值越低，保温性能越好，冬季热损失越少",
                "related_concept": ["U值", "传热系数", "保温"],
            },
            "Visible_Transmittance": {
                "description": "可见光透射率：可见光进入室内的比例",
                "unit": "无量纲 (0-1)",
                "range": [0, 1],
                "semantic": "值越高，自然采光越强，但可能增加眩光和部分太阳得热",
                "related_concept": ["透光", "可见光", "采光"],
            },
        },
        "Lights": {
            "Watts_per_Zone_Floor_Area": {
                "description": "单位楼板面积的照明功率密度",
                "unit": "W/m²",
                "range": [0, 50],
                "semantic": "值越高，照明越亮，耗电越多",
                "related_concept": ["照明密度", "功率密度"],
            },
            "Watts_per_Floor_Area": {
                "description": "单位楼板面积的照明功率密度",
                "unit": "W/m²",
                "range": [0, 50],
                "semantic": "值越高，照明越亮，耗电越多",
                "related_concept": ["照明密度", "功率密度"],
            },
            "Watts_per_Person": {
                "description": "人均照明功率",
                "unit": "W/person",
                "range": [0, 100],
                "semantic": "值越高，单个人需要的照明功率越大",
                "related_concept": ["人均功率"],
            },
            "Lighting_Level": {
                "description": "目标照度（亮度水平）",
                "unit": "lux",
                "range": [0, 2000],
                "semantic": "值越高，照度越高，通常需要更多功率",
                "related_concept": ["照度", "亮度"],
            },
        },
        "OtherEquipment": {
            "Power_per_Zone_Floor_Area": {
                "description": "单位楼板面积的设备功率密度",
                "unit": "W/m²",
                "range": [0, 100],
                "semantic": "值越高，设备散热和耗电越多",
                "related_concept": ["设备功率密度", "设备负荷"],
            },
            "Power_per_Person": {
                "description": "人均设备功率",
                "unit": "W/person",
                "range": [0, 200],
                "semantic": "值越高，单个人相关的设备负荷越大",
                "related_concept": ["人均功率", "设备负荷"],
            },
            "Design_Level": {
                "description": "设备设计功率",
                "unit": "W",
                "range": [0, 100000],
                "semantic": "值越高，设备基础负荷越大",
                "related_concept": ["设备功率", "设计水平"],
            },
        },
        "ZoneInfiltration:DesignFlowRate": {
            "Air_Changes_per_Hour": {
                "description": "换气次数：每小时房间空气被完全交换的次数",
                "unit": "次/小时",
                "range": [0, 10],
                "semantic": "值越高，渗透越严重，冬季热损失越大",
                "related_concept": ["换气", "渗透率", "ACH"],
            },
            "Flow_per_Exterior_Surface_Area": {
                "description": "单位外表面积的渗透流量",
                "unit": "m³/s-m²",
                "range": [0, 0.01],
                "semantic": "值越高，通过单位外墙面积的漏风越多",
                "related_concept": ["渗透", "外墙面积渗透"],
            },
            "Flow_Rate_per_Exterior_Surface_Area": {
                "description": "单位外表面积的渗透流量",
                "unit": "m³/s-m²",
                "range": [0, 0.01],
                "semantic": "值越高，通过单位外墙面积的漏风越多",
                "related_concept": ["渗透", "外墙面积渗透"],
            },
            "Flow_per_Zone_Floor_Area": {
                "description": "单位楼板面积的渗透流量",
                "unit": "m³/s-m²",
                "range": [0, 0.01],
                "semantic": "值越高，按楼板面积折算的漏风越多",
                "related_concept": ["渗透", "面积渗透"],
            },
            "Flow_Rate_per_Floor_Area": {
                "description": "单位楼板面积的渗透流量",
                "unit": "m³/s-m²",
                "range": [0, 0.01],
                "semantic": "值越高，按楼板面积折算的漏风越多",
                "related_concept": ["渗透", "面积渗透"],
            },
        },
        "People": {
            "People_per_Zone_Floor_Area": {
                "description": "人员密度：单位楼板面积的人数",
                "unit": "人/m²",
                "range": [0, 0.2],
                "semantic": "值越低，人员产热负荷越小，制冷能耗降低",
                "related_concept": ["人员密度", "占用率", "灵活办公"],
            },
            "People_per_Floor_Area": {
                "description": "人员密度：单位楼板面积的人数",
                "unit": "人/m²",
                "range": [0, 0.2],
                "semantic": "值越低，人员产热负荷越小，制冷能耗降低",
                "related_concept": ["人员密度", "占用率", "灵活办公"],
            },
            "Zone_Floor_Area_per_Person": {
                "description": "人均楼板面积",
                "unit": "m²/人",
                "range": [5, 50],
                "semantic": "值越高，单位面积人数越少，人员负荷越低",
                "related_concept": ["人均面积", "办公空间"],
            },
            "Floor_Area_per_Person": {
                "description": "人均楼板面积",
                "unit": "m²/人",
                "range": [5, 50],
                "semantic": "值越高，单位面积人数越少，人员负荷越低",
                "related_concept": ["人均面积", "办公空间"],
            },
        },
        "ThermostatSetpoint:DualSetpoint": {
            "Heating_Setpoint_Temperature_Schedule_Name": {
                "description": "供暖设定温度时间表：定义供暖目标温度",
                "unit": "°C",
                "range": [15, 23],
                "semantic": "温度越低，供暖能耗越低（冬季每降低1°C可节省8-15%供暖能耗）",
                "related_concept": ["供暖温度", "冬季设定点", "舒适温度"],
                "optimization_note": "建议从20-22°C降至18-19°C",
            },
            "Cooling_Setpoint_Temperature_Schedule_Name": {
                "description": "制冷设定温度时间表：定义制冷目标温度",
                "unit": "°C",
                "range": [24, 28],
                "semantic": "温度越高，制冷能耗越低（夏季每提高1°C可节省8-12%制冷能耗）",
                "related_concept": ["制冷温度", "夏季设定点", "空调温度"],
                "optimization_note": "建议从24-25°C提升至25-26°C",
            },
        },
        "DesignSpecification:OutdoorAir": {
            "Outdoor_Air_Flow_per_Person": {
                "description": "人均新风量：每人所需的室外新风流量",
                "unit": "m³/s/人",
                "range": [0.005, 0.015],
                "semantic": "值越小，新风负荷越低，但需满足健康标准（GB/T18883建议≥0.008）",
                "related_concept": ["人均新风", "通风率", "室内空气质量"],
                "optimization_note": "办公室标准0.008-0.010 m³/s/人",
            },
            "Outdoor_Air_Flow_per_Zone_Floor_Area": {
                "description": "单位楼板面积新风量",
                "unit": "m³/s/m²",
                "range": [0, 0.002],
                "semantic": "值越小，新风负荷越低",
                "related_concept": ["新风密度", "面积新风"],
            },
            "Outdoor_Air_Flow_Air_Changes_per_Hour": {
                "description": "按换气次数表示的新风量",
                "unit": "次/小时",
                "range": [0, 10],
                "semantic": "值越小，新风负荷越低，但需结合健康和渗透控制综合判断",
                "related_concept": ["换气次数", "ACH", "通风"],
            },
        },
        "DesignSpecification:ZoneAirDistribution": {
            "Zone_Air_Distribution_Effectiveness_in_Cooling_Mode": {
                "description": "制冷工况空气分配效率",
                "unit": "无量纲",
                "range": [0.5, 1.5],
                "semantic": "值越高，达到同等舒适所需的新风/送风量可降低，潜在减少空调负荷",
                "related_concept": ["送风效率", "通风效率"],
            },
            "Zone_Air_Distribution_Effectiveness_in_Heating_Mode": {
                "description": "供暖工况空气分配效率",
                "unit": "无量纲",
                "range": [0.5, 1.5],
                "semantic": "值越高，供暖气流利用更充分，潜在降低供暖负荷",
                "related_concept": ["送风效率", "通风效率"],
            },
        },
        "Daylighting:Controls": {
            "Minimum_Input_Power_Fraction_for_Continuous_or_ContinuousOff_Dimming_Control": {
                "description": "连续调光时照明系统最小输入功率比例",
                "unit": "无量纲 (0-1)",
                "range": [0, 1],
                "semantic": "值越低，日光充足时照明可降得更低，照明能耗下降潜力更大",
                "related_concept": ["调光", "日光控制", "照明节能"],
            },
            "Minimum_Light_Output_Fraction_for_Continuous_or_ContinuousOff_Dimming_Control": {
                "description": "连续调光时照明系统最小出光比例",
                "unit": "无量纲 (0-1)",
                "range": [0, 1],
                "semantic": "值越低，允许更深度调光，日光利用率更高",
                "related_concept": ["调光", "照明输出"],
            },
            "Illuminance_Setpoint_at_Reference_Point_1": {
                "description": "参考点1照度设定值",
                "unit": "lux",
                "range": [100, 1000],
                "semantic": "值越低，照明系统目标照度越低，照明能耗通常下降",
                "related_concept": ["照度", "设定点", "日光控制"],
            },
            "Illuminance_Setpoint_at_Reference_Point_2": {
                "description": "参考点2照度设定值",
                "unit": "lux",
                "range": [100, 1000],
                "semantic": "值越低，照明系统目标照度越低，照明能耗通常下降",
                "related_concept": ["照度", "设定点", "日光控制"],
            },
        },
        "ZoneHVAC:IdealLoadsAirSystem": {
            "Minimum_Cooling_Supply_Air_Temperature": {
                "description": "最小制冷供风温度阈值：运行时供冷送风温度下限",
                "unit": "°C",
                "range": [12, 20],
                "semantic": "值越高，送风下限越温和，通常可降低制冷侧能耗",
                "related_concept": ["阈值", "运行限制", "最小制冷供风温度"],
                "optimization_note": "常见可从16°C提升至17-18°C",
            },
            "Maximum_Heating_Supply_Air_Temperature": {
                "description": "最大供暖供风温度阈值：运行时供暖送风温度上限",
                "unit": "°C",
                "range": [35, 60],
                "semantic": "值越低，供暖送风峰值受限，通常可降低供暖侧峰值能耗",
                "related_concept": ["阈值", "运行限制", "最大供暖供风温度"],
                "optimization_note": "建议在舒适度允许范围内小幅下调",
            },
            "Sensible_Heat_Recovery_Effectiveness": {
                "description": "显热回收效率",
                "unit": "无量纲 (0-1)",
                "range": [0, 1],
                "semantic": "值越高，排风显热回收越充分，供暖/制冷负荷都可下降",
                "related_concept": ["热回收", "显热效率"],
            },
            "Latent_Heat_Recovery_Effectiveness": {
                "description": "潜热回收效率",
                "unit": "无量纲 (0-1)",
                "range": [0, 1],
                "semantic": "值越高，湿负荷回收越充分，可降低空调系统潜热负荷",
                "related_concept": ["热回收", "潜热效率"],
            },
            "Minimum_Cooling_Supply_Air_Humidity_Ratio": {
                "description": "最小制冷供风含湿量下限",
                "unit": "kgWater/kgDryAir",
                "range": [0.005, 0.012],
                "semantic": "值越低，除湿能力越强但潜热处理能耗可能上升",
                "related_concept": ["含湿量", "湿度比", "除湿"],
            },
            "Maximum_Heating_Supply_Air_Humidity_Ratio": {
                "description": "最大供暖供风含湿量上限",
                "unit": "kgWater/kgDryAir",
                "range": [0.005, 0.02],
                "semantic": "值越高，供暖加湿潜力更大，但可能增加湿负荷处理成本",
                "related_concept": ["含湿量", "湿度比", "加湿"],
            },
        },
        "Sizing:Zone": {
            "Zone_Cooling_Design_Supply_Air_Temperature": {
                "description": "制冷设计供风温度：空调送风温度",
                "unit": "°C",
                "range": [12, 18],
                "semantic": "温度越高，制冷机组效率越高，能耗越低（但需保证除湿能力）",
                "related_concept": ["供风温度", "送风温度", "制冷供风"],
                "optimization_note": "可从13-14°C提升至15-17°C",
            },
            "Zone_Heating_Design_Supply_Air_Temperature": {
                "description": "供暖设计供风温度：暖气送风温度",
                "unit": "°C",
                "range": [35, 55],
                "semantic": "温度越低，供暖效率越高（热泵COP提升），能耗越低",
                "related_concept": ["供风温度", "供暖送风"],
                "optimization_note": "可从50°C降至42-45°C（地暖30-35°C）",
            },
            "Zone_Cooling_Design_Supply_Air_Humidity_Ratio": {
                "description": "制冷设计供风含湿量",
                "unit": "kgWater/kgDryAir",
                "range": [0.005, 0.012],
                "semantic": "值越低，设计除湿要求越高，潜热处理能力增强但能耗可能增加",
                "related_concept": ["含湿量", "湿度比", "除湿设计"],
            },
            "Zone_Heating_Design_Supply_Air_Humidity_Ratio": {
                "description": "供暖设计供风含湿量",
                "unit": "kgWater/kgDryAir",
                "range": [0.005, 0.02],
                "semantic": "影响供暖季空气湿度目标，过高可能增加处理能耗",
                "related_concept": ["含湿量", "湿度比", "加湿设计"],
            },
            "Zone_Cooling_Sizing_Factor": {
                "description": "制冷设计裕量系数",
                "unit": "无量纲",
                "range": [1.0, 1.5],
                "semantic": "值越高，系统偏保守，设备容量和峰值能耗潜力更高",
                "related_concept": ["裕量", "安全系数", "过度设计"],
            },
            "Zone_Heating_Sizing_Factor": {
                "description": "供暖设计裕量系数",
                "unit": "无量纲",
                "range": [1.0, 1.5],
                "semantic": "值越高，系统偏保守，供暖容量和峰值负荷潜力更高",
                "related_concept": ["裕量", "安全系数", "过度设计"],
            },
        },
    }


def analyze_user_intent(user_request: str) -> Dict:
    """
    分析用户的真实意图（用户可能不了解IDF结构）。
    比如"降低遮阳系数"可能意味着"增加反射率"或"减少透射率"。
    """
    intent = {
        'raw_request': user_request,
        'likely_goals': [],
        'possible_actions': [],
    }

    request_lower = user_request.lower()

    user_intent_patterns = {
        '降低.*太阳.*辐射': {
            'goals': ['减少太阳热进入室内', '提高夏季隔热能力'],
            'actions': ['增加反射率', '减少透射率', '提高SHGC'],
        },
        '减少.*遮阳': {
            'goals': ['减少遮阳效果', '增加光透入'],
            'actions': ['降低反射率', '提高透射率', '降低SHGC'],
        },
        '降低.*遮阳': {
            'goals': ['减少太阳热进入', '增加遮阳效果'],
            'actions': ['增加反射率', '减少透射率', '降低SHGC'],
        },
        '提高.*隔热': {
            'goals': ['增加隔热性能', '减少热传递'],
            'actions': ['减少透射率', '增加反射率', '增加厚度'],
        },
        '改善.*保温': {
            'goals': ['提高保温性能', '减少热损失'],
            'actions': ['增加热阻', '降低导热系数', '增加厚度'],
        },
        '降低.*渗透': {
            'goals': ['减少冬季热损失', '提高气密性'],
            'actions': ['减少换气次数', '减少单位面积渗透量'],
        },
        '减少.*空气.*渗透': {
            'goals': ['减少冬季热损失', '提高气密性'],
            'actions': ['减少换气次数', '减少单位面积渗透量'],
        },
        '降低.*照明': {
            'goals': ['减少照明能耗', '降低照明功率'],
            'actions': ['降低照明功率密度', '降低人均照明功率'],
        },
        '提高.*照明.*效率': {
            'goals': ['提高照明效率', '在相同亮度下减少功耗'],
            'actions': ['降低照明功率密度'],
        },
    }

    for pattern, analysis in user_intent_patterns.items():
        if re.search(pattern, request_lower):
            intent['likely_goals'] = analysis.get('goals', [])
            intent['possible_actions'] = analysis.get('actions', [])
            break

    if not intent['likely_goals']:
        keyword_intent_map = {
            '降低': {
                'goals': ['降低该参数的值或影响'],
                'actions': ['减少相关字段值', '提高相关阻力'],
            },
            '提高': {
                'goals': ['提高该参数的值或影响'],
                'actions': ['增加相关字段值', '降低相关阻力'],
            },
            '减少': {
                'goals': ['减少相关影响'],
                'actions': ['降低相关参数值'],
            },
        }

        for keyword, analysis in keyword_intent_map.items():
            if keyword in request_lower:
                intent['likely_goals'] = analysis.get('goals', [])
                intent['possible_actions'] = analysis.get('actions', [])
                break

    return intent
