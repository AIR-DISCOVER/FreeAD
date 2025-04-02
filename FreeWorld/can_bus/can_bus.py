import json
from typing import List, Dict, Optional
from bisect import bisect_right

class CanBusData:
    """
    自定义 CAN 总线数据管理类，读取 can_bus.json 并提供数据访问接口。
    """
    def __init__(self, json_path: str):
        """
        初始化 RobotCanBus 实例，读取并组织 JSON 数据。
        
        参数:
            json_path (str): can_bus.json 文件的路径。
        """
        self.data = self._load_json(json_path)
        self._sort_messages()

    def _load_json(self, json_path: str) -> Dict[str, List[Dict]]:
        """
        读取 JSON 文件并返回数据字典。
        
        参数:
            json_path (str): JSON 文件路径。
        
        返回:
            Dict[str, List[Dict]]: 按场景名称组织的消息列表。
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading JSON file {json_path}: {e}")
            return {}

    def _sort_messages(self):
        """
        对每个场景的消息按 utime 进行排序，确保消息按时间顺序排列。
        """
        for scene, scene_data in self.data.items():
            if 'data' in scene_data:
                scene_data['data'].sort(key=lambda msg: msg['utime'])

    def get_messages(self, scene_name: str, msg_type: str) -> List[Dict]:
        """
        获取指定场景中指定类型的所有消息。
        
        参数:
            scene_name (str): 场景名称。
            msg_type (str): 消息类型，如 'pose' 或 'steer'。
        
        返回:
            List[Dict]: 指定类型的消息列表。
        """
        scene = self.data.get(scene_name, None)
        if not scene or 'data' not in scene or not scene['data']:
            print(f"Warning: No data found for scene {scene_name}.")
            return []
        
        # 过滤出指定类型的消息
        return [msg for msg in scene['data'] if msg.get('type') == msg_type]

    def get_latest_before(self, scene_name: str, timestamp: int) -> Optional[Dict]:
        """
        获取指定场景中，时间戳小于等于给定 timestamp 的最新消息。
        
        参数:
            scene_name (str): 场景名称。
            timestamp (int): 时间戳（微秒）。
        
        返回:
            Optional[Dict]: 最新的消息，如果不存在则返回 None。
        """
        scene = self.data.get(scene_name, None)
        if not scene or 'data' not in scene or not scene['data']:
            print(f"Warning: No data found for scene {scene_name}.")
            return None

        # 使用二分查找优化查找过程，寻找跟时间戳最匹配的数据
        utimes = [msg['utime'] for msg in scene['data']]
        # 使用 bisect_right 查找第一个大于 timestamp 的位置
        index = bisect_right(utimes, timestamp) - 1

        if index >= 0:
            # 确保找到的时间戳小于等于给定的 timestamp
            if utimes[index] <= timestamp:
                return scene['data'][index]
            else:
                print(f"Warning: No valid can_bus messages found for timestamp {timestamp} in scene {scene_name}.")
                return None
        else:
            print(f"Warning: No can_bus messages found for timestamp {timestamp} in scene {scene_name}.")
            return None
