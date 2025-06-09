from pathlib import Path
from typing import (
    Any,
    Dict,
    Hashable
)
from binstorage import BinaryStorage


class Namer:
    NOT_FOUND: int = -1
    def __init__(self, name: int, storage_path: Path):
        full_path = storage_path / f'namer_{name}.bin'
        default_storage: Dict[str, Any] = {
            'vertex_map': {},
            'next_vertex_name': 0
        }
        self.__storage = BinaryStorage(full_path, default_storage)
        self.__vertex_map: Dict[Hashable, int] = self.__storage.data.get('vertex_map', default_storage['vertex_map'])
        self.__next_vertex_name: int = self.__storage.data.get('next_vertex_name', default_storage['next_vertex_name'])
    
    def name_it(self, vertex: Hashable | None) -> int:
        if vertex in self.__vertex_map:
            return self.__vertex_map.get(vertex)
        
        v_id = self.__next_vertex_name
        self.__vertex_map[vertex] = v_id
        self.__next_vertex_name += 1
        return v_id
    
    def unname_it(self, name: int) -> Hashable:
        return next((v for v, n in self.__vertex_map.items() if n == name), self.NOT_FOUND)
    
    def dump(self):
        self.__storage.dump()
    
    
if __name__ == '__main__':
    namer = Namer(0, Path('./datasets/early/namer'))
    v1 = namer.name_it(None)
    v2 = namer.name_it(0)
    v3 = namer.name_it('a')
    v4 = namer.name_it((0, 1, 2))
    print(v1, v2, v3, v4)
    
    v4u = namer.unname_it(5)
    print(v4u)
    namer.dump()