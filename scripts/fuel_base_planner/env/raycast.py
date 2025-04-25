import numpy as np
import math
from typing import List, Optional, Tuple, Union
from ..utils import Vector3d, Vector3i

def signum(x: Union[int, float]) -> Union[int, float]:
    if x == 0:
        return 0
    return -1 if x < 0 else 1

def mod(value: float, modulus: float) -> float:
    return math.fmod(math.fmod(value, modulus) + modulus, modulus)

def intbound(s: float, ds: float) -> float:
    # Find the smallest positive t such that s+t*ds is an integer
    if ds < 0:
        return intbound(-s, -ds)
    else:
        s = mod(s, 1)
        # problem is now s+t*ds = 1
        return (1 - s) / ds

def raycast(start: Vector3d, end: Vector3d, min_bound: Vector3d, 
           max_bound: Vector3d, output_points_cnt: int, output: List[Vector3d]) -> List[Vector3d]:
    x = int(math.floor(start.x))
    y = int(math.floor(start.y))
    z = int(math.floor(start.z))
    endX = int(math.floor(end.x))
    endY = int(math.floor(end.y))
    endZ = int(math.floor(end.z))
    direction = end - start
    maxDist = direction.norm()

    # Break out direction vector
    dx = endX - x
    dy = endY - y 
    dz = endZ - z

    # Direction to increment x,y,z when stepping
    stepX = signum(dx)
    stepY = signum(dy)
    stepZ = signum(dz)

    # Initial values depend on fractional part of origin
    tMaxX = intbound(start.x, dx)
    tMaxY = intbound(start.y, dy)
    tMaxZ = intbound(start.z, dz)

    # Change in t when taking a step (always positive)
    tDeltaX = stepX / dx if dx != 0 else float('inf')
    tDeltaY = stepY / dy if dy != 0 else float('inf')
    tDeltaZ = stepZ / dz if dz != 0 else float('inf')

    # Avoid infinite loop
    if stepX == 0 and stepY == 0 and stepZ == 0:
        return output

    dist = 0
    while True:
        if (x >= min_bound.x and x < max_bound.x and 
            y >= min_bound.y and y < max_bound.y and
            z >= min_bound.z and z < max_bound.z):
            
            output[output_points_cnt] = Vector3d(x, y, z)
            output_points_cnt += 1

            dist = math.sqrt((x - start.x)**2 + (y - start.y)**2 + (z - start.z)**2)
            if dist > maxDist:
                return output

        if x == endX and y == endY and z == endZ:
            break

        # Choose closest cube boundary
        if tMaxX < tMaxY:
            if tMaxX < tMaxZ:
                x += stepX
                tMaxX += tDeltaX
            else:
                z += stepZ
                tMaxZ += tDeltaZ
        else:
            if tMaxY < tMaxZ:
                y += stepY
                tMaxY += tDeltaY
            else:
                z += stepZ
                tMaxZ += tDeltaZ
    
    return output

def raycast_vector(start: Vector3d, end: Vector3d, min_bound: Vector3d,
                  max_bound: Vector3d, output: List[Vector3d]) -> List[Vector3d]:
    x = int(math.floor(start.x))
    y = int(math.floor(start.y))
    z = int(math.floor(start.z))
    endX = int(math.floor(end.x))
    endY = int(math.floor(end.y))
    endZ = int(math.floor(end.z))
    direction = end - start
    maxDist = direction.norm()

    # Break out direction vector
    dx = endX - x
    dy = endY - y 
    dz = endZ - z

    # Direction to increment x,y,z when stepping
    stepX = signum(dx)
    stepY = signum(dy)
    stepZ = signum(dz)

    # Initial values depend on fractional part of origin
    tMaxX = intbound(start.x, dx)
    tMaxY = intbound(start.y, dy)
    tMaxZ = intbound(start.z, dz)

    # Change in t when taking a step (always positive)
    tDeltaX = stepX / dx if dx != 0 else float('inf')
    tDeltaY = stepY / dy if dy != 0 else float('inf')
    tDeltaZ = stepZ / dz if dz != 0 else float('inf')

    output.clear()

    # Avoid infinite loop
    if stepX == 0 and stepY == 0 and stepZ == 0:
        return output

    dist = 0
    while True:
        if (x >= min_bound.x and x < max_bound.x and 
            y >= min_bound.y and y < max_bound.y and
            z >= min_bound.z and z < max_bound.z):
            
            output.append(Vector3d(x, y, z))
            
            dist = (Vector3d(x, y, z) - start).norm()
            
            if dist > maxDist:
                return output

            if len(output) > 1500:
                raise ValueError("Error, too many raycast voxels")

        if x == endX and y == endY and z == endZ:
            break

        # Choose closest cube boundary
        if tMaxX < tMaxY:
            if tMaxX < tMaxZ:
                x += stepX
                tMaxX += tDeltaX
            else:
                z += stepZ
                tMaxZ += tDeltaZ
        else:
            if tMaxY < tMaxZ:
                y += stepY
                tMaxY += tDeltaY
            else:
                z += stepZ
                tMaxZ += tDeltaZ

    return output

class RayCaster:
    def __init__(self):
        self.start_ = Vector3d()
        self.end_ = Vector3d()
        self.direction_ = Vector3d()
        self.min_ = Vector3d()
        self.max_ = Vector3d()
        self.x_ = 0
        self.y_ = 0
        self.z_ = 0
        self.endX_ = 0
        self.endY_ = 0
        self.endZ_ = 0
        self.maxDist_ = 0.0
        self.dx_ = 0.0
        self.dy_ = 0.0
        self.dz_ = 0.0
        self.stepX_ = 0
        self.stepY_ = 0
        self.stepZ_ = 0
        self.tMaxX_ = 0.0
        self.tMaxY_ = 0.0
        self.tMaxZ_ = 0.0
        self.tDeltaX_ = 0.0
        self.tDeltaY_ = 0.0
        self.tDeltaZ_ = 0.0
        self.dist_ = 0.0
        self.step_num_ = 0
        self.resolution_ = 0.0
        self.offset_ = Vector3d()
        self.half_ = Vector3d()

    def setParams(self, res: float, origin: Vector3d) -> None:
        self.resolution_ = res
        self.half_ = Vector3d(0.5, 0.5, 0.5)
        self.offset_ = self.half_ - origin / res

    def input(self, start: Vector3d, end: Vector3d) -> bool:
        
        start = start / self.resolution_
        end = end / self.resolution_

        self.x_ = int(math.floor(start.x))
        self.y_ = int(math.floor(start.y))
        self.z_ = int(math.floor(start.z))

        self.endX_ = int(math.floor(end.x))
        self.endY_ = int(math.floor(end.y))
        self.endZ_ = int(math.floor(end.z))

        self.direction_ = end - start
        self.maxDist_ = self.direction_.norm()

        self.dx_ = self.endX_ - self.x_
        self.dy_ = self.endY_ - self.y_
        self.dz_ = self.endZ_ - self.z_

        self.stepX_ = signum(self.dx_)
        self.stepY_ = signum(self.dy_)
        self.stepZ_ = signum(self.dz_)

        self.tMaxX_ = intbound(self.x_, self.dx_)
        self.tMaxY_ = intbound(self.y_, self.dy_)
        self.tMaxZ_ = intbound(self.z_, self.dz_)

        self.tDeltaX_ = self.stepX_ / self.dx_
        self.tDeltaY_ = self.stepY_ / self.dy_
        self.tDeltaZ_ = self.stepZ_ / self.dz_

        self.dist_ = 0.0
        self.step_num_ = 0

        if(self.stepX_ == 0 and self.stepY_ == 0 and self.stepZ_ == 0):
            return False
        else:
            return True              

    def nextId(self, idx: Vector3i) -> bool:
        
        tmp = Vector3d(self.x_, self.y_, self.z_)
        idx.x = int(tmp.x + self.offset_.x)
        idx.y = int(tmp.y + self.offset_.y) 
        idx.z = int(tmp.z + self.offset_.z)

        if self.x_ == self.endX_ and self.y_ == self.endY_ and self.z_ == self.endZ_:
            return False

        if self.tMaxX_ < self.tMaxY_:
            if self.tMaxX_ < self.tMaxZ_:
                self.x_ += self.stepX_
                self.tMaxX_ += self.tDeltaX_
            else:
                self.z_ += self.stepZ_
                self.tMaxZ_ += self.tDeltaZ_
        else:
            if self.tMaxY_ < self.tMaxZ_:
                self.y_ += self.stepY_
                self.tMaxY_ += self.tDeltaY_
            else:
                self.z_ += self.stepZ_
                self.tMaxZ_ += self.tDeltaZ_

        return True

    def nextPos(self, pos: Vector3d) -> bool:
        
        tmp = Vector3d(self.x_, self.y_, self.z_)
        pos = Vector3d(tmp.x + self.half_.x, tmp.y + self.half_.y, tmp.z + self.half_.z)
        pos *= self.resolution_

        if self.x_ == self.endX_ and self.y_ == self.endY_ and self.z_ == self.endZ_:
            return False

        if self.tMaxX_ < self.tMaxY_:
            if self.tMaxX_ < self.tMaxZ_:
                self.x_ += self.stepX_
                self.tMaxX_ += self.tDeltaX_
            else:
                self.z_ += self.stepZ_
                self.tMaxZ_ += self.tDeltaZ_
        else:
            if self.tMaxY_ < self.tMaxZ_:
                self.y_ += self.stepY_
                self.tMaxY_ += self.tDeltaY_
            else:
                self.z_ += self.stepZ_
                self.tMaxZ_ += self.tDeltaZ_

        return True

    def setInput(self, start: Vector3d, end: Vector3d) -> bool:
        self.start_ = start
        self.end_ = end

        self.x_ = int(math.floor(start.x))
        self.y_ = int(math.floor(start.y)) 
        self.z_ = int(math.floor(start.z))
        self.endX_ = int(math.floor(end.x))
        self.endY_ = int(math.floor(end.y))
        self.endZ_ = int(math.floor(end.z))
        self.direction_ = end - start
        self.maxDist_ = self.direction_.norm()

        # Break out direction vector
        self.dx_ = self.endX_ - self.x_
        self.dy_ = self.endY_ - self.y_
        self.dz_ = self.endZ_ - self.z_

        # Direction to increment x,y,z when stepping
        self.stepX_ = int(np.sign(self.dx_))
        self.stepY_ = int(np.sign(self.dy_))
        self.stepZ_ = int(np.sign(self.dz_))

        # Initial values depend on fractional part of origin
        self.tMaxX_ = intbound(start.x, self.dx_)
        self.tMaxY_ = intbound(start.y, self.dy_)
        self.tMaxZ_ = intbound(start.z, self.dz_)

        # Change in t when taking a step (always positive)
        self.tDeltaX_ = float(self.stepX_) / self.dx_ if self.dx_ != 0 else float('inf')
        self.tDeltaY_ = float(self.stepY_) / self.dy_ if self.dy_ != 0 else float('inf')
        self.tDeltaZ_ = float(self.stepZ_) / self.dz_ if self.dz_ != 0 else float('inf')

        self.dist_ = 0
        self.step_num_ = 0

        # Avoid infinite loop
        if self.stepX_ == 0 and self.stepY_ == 0 and self.stepZ_ == 0:
            return False
        return True

    def step(self, ray_pt: Vector3d) -> bool:
        
        ray_pt = Vector3d(self.x_, self.y_, self.z_)

        if self.x_ == self.endX_ and self.y_ == self.endY_ and self.z_ == self.endZ_:
            return False

        if self.tMaxX_ < self.tMaxY_:
            if self.tMaxX_ < self.tMaxZ_:
                self.x_ += self.stepX_
                self.tMaxX_ += self.tDeltaX_
            else:
                self.z_ += self.stepZ_
                self.tMaxZ_ += self.tDeltaZ_
        else:
            if self.tMaxY_ < self.tMaxZ_:
                self.y_ += self.stepY_
                self.tMaxY_ += self.tDeltaY_
            else:
                self.z_ += self.stepZ_
                self.tMaxZ_ += self.tDeltaZ_

        return True
