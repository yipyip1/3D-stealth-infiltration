BULLET_SPEED = 18.0
BULLET_LIFETIME = 50
FIRE_COOLDOWN = 8                        

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
import random
import time

                               
W, H = 1200, 900
GRID_SIZE = 40
GRID_TILES = 21
MAZE_SIZE = GRID_SIZE * GRID_TILES // 2

                   
MOVE_SPEED = 3.0
PLAYER_MAX_HP = 100
PLAYER_RADIUS = 13

        
CAMERA_DISTANCE = 140
CAMERA_MIN_DISTANCE = 60
CAMERA_HEIGHT = 500
CAMERA_SMOOTH = 0.15
CAM_RADIUS = 12
THIRD_PERSON_CAM_POS = [0, 500, 500]
CONE_LENGTH = 80
CONE_ANGLE = 30

       
MAZE_WALL_THICKNESS = 8
WALL_THICKNESS = 12

                 
OUTSIDE_SIZE = 600
HOUSE_POS = (0, 0, 0)
HOUSE_SIZE = 200

              
TARGET_FPS = 60
FRAME_TIME = 1000 // TARGET_FPS                          
KEY_HOLD_MS = 200                                                          

       
ENEMY_RADIUS = 15
ENEMY_HEIGHT = 60
ENEMY_HP = 3
ENEMY_DETECTION_RADIUS = 200
ENEMY_ATTACK_RADIUS = 150
ENEMY_SPEED = 2.0
ENEMY_DAMAGE = 10

                           
            
                           
GAME_OUTSIDE = 0
GAME_INSIDE = 1
GAME_WIN = 2
GAME_LOSE = 3

game_state = GAME_OUTSIDE
player_pos = [0.0, -OUTSIDE_SIZE + 50, 0.0]
player_angle = 90.0
player_hp = PLAYER_MAX_HP
player_ammo = 30
player_weapon_damage = 1
player_fire_rate = 1.0              

camera_pos = [0.0, 0.0, CAMERA_HEIGHT]
camera_mode = 0                                                                                     

detection_level = 0.0
detection_meter = 0.0                           
detected = False                         
seen_frames = 0                                                                    
game_won = False
game_lost = False
debug_mode = False

                          
current_level = 1              
exit_door_open = False                                   

                                                                          
LEVEL_DEFS = {
    1: {'enemy_count': 3, 'enemy_hp': 3, 'enemy_damage': 10, 'detection_radius': 200},
    2: {'enemy_count': 4, 'enemy_hp': 4, 'enemy_damage': 12, 'detection_radius': 220},
    3: {'enemy_count': 5, 'enemy_hp': 5, 'enemy_damage': 15, 'detection_radius': 250},
}

                                                                    
LEVEL_LAYOUTS = {
    1: [
        {'area': (-MAZE_SIZE, -MAZE_SIZE, -MAZE_SIZE//2, MAZE_SIZE//2), 'door': (-MAZE_SIZE//2, -MAZE_SIZE//2)},
        {'area': (-MAZE_SIZE//2, -MAZE_SIZE, MAZE_SIZE//2, -MAZE_SIZE//2), 'door': (0, -MAZE_SIZE//2)},
        {'area': (MAZE_SIZE//2, -MAZE_SIZE//2, MAZE_SIZE, MAZE_SIZE//2), 'door': (MAZE_SIZE//2, 0)},
        {'area': (-MAZE_SIZE//2, MAZE_SIZE//2, MAZE_SIZE//2, MAZE_SIZE), 'door': (0, MAZE_SIZE//2)},
    ],
    2: [
        {'area': (-MAZE_SIZE, -MAZE_SIZE, -MAZE_SIZE//2, MAZE_SIZE//2), 'door': (-MAZE_SIZE//2, -MAZE_SIZE//2)},
        {'area': (-MAZE_SIZE//2, -MAZE_SIZE, MAZE_SIZE//2, -MAZE_SIZE//2), 'door': (0, -MAZE_SIZE//2)},
        {'area': (-MAZE_SIZE//2, -MAZE_SIZE//2, MAZE_SIZE//2, MAZE_SIZE//2), 'door': (MAZE_SIZE//2, 0)},
        {'area': (MAZE_SIZE//2, -MAZE_SIZE//2, MAZE_SIZE, MAZE_SIZE//2), 'door': (MAZE_SIZE//2, MAZE_SIZE//2)},
    ],
    3: [
        {'area': (-MAZE_SIZE, -MAZE_SIZE, -MAZE_SIZE//2, 0), 'door': (-MAZE_SIZE//2, -MAZE_SIZE//2)},
        {'area': (-MAZE_SIZE//2, -MAZE_SIZE, MAZE_SIZE//2, -MAZE_SIZE//2), 'door': (0, -MAZE_SIZE//2)},
        {'area': ( -MAZE_SIZE//2, 0, MAZE_SIZE//2, MAZE_SIZE//2), 'door': (0, MAZE_SIZE//2)},
        {'area': (MAZE_SIZE//2, -MAZE_SIZE//2, MAZE_SIZE, MAZE_SIZE//2), 'door': (MAZE_SIZE//2, 0)},
    ]
}

                                     
corridors = []                                   

                                                 
player_weapon_state = 0
player_powerups = {'knife': False, 'gun': False, 'invisibility': False, 'distraction': False}
player_gun_ammo = 0
player_knife_hits = {}
player_barehands_hits = {}
invisibility_timer = 0.0
invisibility_active = False
distraction = None
player_collision_damage_cooldown = 0

              
player_lives = 13

       
paused = False

fire_cooldown = 0
animation_time = 0.0
player_moved_this_frame = False
last_update_time = 0

                                             
bullets = []

                                                                        
                          
enemies = []

                                  
                                     
pickups = []

                              
maze_grid = []
walls = []

               
trees = []
fence_posts = []
rocks = []                           

               
front_door_trigger = None
exit_door_trigger = None

                
keys_pressed = set()
key_timestamps = {}

                          
inside_static_list = None
quadric = gluNewQuadric()
last_game_state = GAME_OUTSIDE                                                        

                           
                 
                           
def generate_maze(width, height):
    maze = [[True for _ in range(width)] for _ in range(height)]
    
    start_x, start_y = 1, 1
    maze[start_y][start_x] = False
    
    stack = [(start_x, start_y)]
    visited = set()
    visited.add((start_x, start_y))
    
    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
    
    while stack:
        cx, cy = stack[-1]
        neighbors = []
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 < nx < width-1 and 0 < ny < height-1:
                if (nx, ny) not in visited:
                    neighbors.append((nx, ny, cx + dx//2, cy + dy//2))
        
        if neighbors:
            nx, ny, mx, my = random.choice(neighbors)
            maze[my][mx] = False
            maze[ny][nx] = False
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()
    
    return maze

def widen_maze(maze):
    height, width = len(maze), len(maze[0])
    widened = [row[:] for row in maze]
                                                                                 
    for y in range(1, height-1):
        for x in range(1, width-1):
            if not maze[y][x]:        
                                                                                 
                if random.random() < 0.25:
                    if random.random() < 0.5:
                        if x+1 < width-1:
                            widened[y][x+1] = False
                    else:
                        if y+1 < height-1:
                            widened[y+1][x] = False

    return widened

def add_rooms(maze, num_rooms=3):
    height, width = len(maze), len(maze[0])

    for _ in range(num_rooms):
        rw, rh = random.randint(3, 5), random.randint(3, 5)
        rx = random.randint(2, max(2, width - rw - 3))
        ry = random.randint(2, max(2, height - rh - 3))

        for y in range(ry, min(ry + rh, height-1)):
            for x in range(rx, min(rx + rw, width-1)):
                maze[y][x] = False

    return maze

def carve_main_path(maze):
    height, width = len(maze), len(maze[0])
    cx = width // 2
    cy = 1
    maze[cy][cx] = False

                                                                                 
    while cy < height - 2:
                                              
        r = random.random()
        if r < 0.65:
            ny, nx = cy + 1, cx
        elif r < 0.825:
            ny, nx = cy, cx + 1
        else:
            ny, nx = cy, cx - 1

                            
        nx = max(1, min(width - 2, nx))
        ny = max(1, min(height - 2, ny))

        cy, cx = ny, nx
        maze[cy][cx] = False
                                                 
        if cx + 1 < width - 1:
            maze[cy][cx + 1] = False
        if cx - 1 > 0:
            maze[cy][cx - 1] = False

                                         
    exit_x = width // 2
    for x in range(exit_x - 1, exit_x + 2):
        if 0 < x < width - 1:
            maze[height - 2][x] = False

    return maze

def carve_entrance_room(maze):
    height, width = len(maze), len(maze[0])
                                                                                       
    for y in range(1, min(3, height-1)):
        for x in range(width//2 - 1, min(width//2 + 2, width-1)):
            maze[y][x] = False
    return maze

def expand_paths(maze, pad=1):
    h, w = len(maze), len(maze[0])
    out = [row[:] for row in maze]
    for y in range(h):
        for x in range(w):
            if not maze[y][x]:
                                            
                for dy in range(-pad, pad+1):
                    for dx in range(-pad, pad+1):
                        ny = y + dy
                        nx = x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            out[ny][nx] = False
    return out

def maze_to_walls(maze):
    height, width = len(maze), len(maze[0])
    offset_x = -MAZE_SIZE
    offset_y = -MAZE_SIZE
    
    edges = []
    
    for y in range(height):
        for x in range(width):
            if maze[y][x]:             
                                               
                if y == 0 or not maze[y-1][x]:
                    wx1 = offset_x + x * GRID_SIZE
                    wy = offset_y + y * GRID_SIZE
                    edges.append((wx1, wy, wx1 + GRID_SIZE, wy, 'H'))
                
                if y == height-1 or not maze[y+1][x]:
                    wx1 = offset_x + x * GRID_SIZE
                    wy = offset_y + (y+1) * GRID_SIZE
                    edges.append((wx1, wy, wx1 + GRID_SIZE, wy, 'H'))
                
                if x == 0 or not maze[y][x-1]:
                    wx = offset_x + x * GRID_SIZE
                    wy1 = offset_y + y * GRID_SIZE
                    edges.append((wx, wy1, wx, wy1 + GRID_SIZE, 'V'))
                
                if x == width-1 or not maze[y][x+1]:
                    wx = offset_x + (x+1) * GRID_SIZE
                    wy1 = offset_y + y * GRID_SIZE
                    edges.append((wx, wy1, wx, wy1 + GRID_SIZE, 'V'))
    
                    
    walls = []
    h_edges = sorted([e for e in edges if e[4] == 'H'], key=lambda e: (e[1], e[0]))
    i = 0
    while i < len(h_edges):
        x1, y, x2, _, _ = h_edges[i]
        j = i + 1
        while j < len(h_edges) and h_edges[j][1] == y and h_edges[j][0] == x2:
            x2 = h_edges[j][2]
            j += 1
        walls.append([x1, y, x2, y, 100])
        i = j
    
    v_edges = sorted([e for e in edges if e[4] == 'V'], key=lambda e: (e[0], e[1]))
    i = 0
    while i < len(v_edges):
        x, y1, _, y2, _ = v_edges[i]
        j = i + 1
        while j < len(v_edges) and v_edges[j][0] == x and v_edges[j][1] == y2:
            y2 = v_edges[j][3]
            j += 1
        walls.append([x, y1, x, y2, 100])
        i = j
    
    return walls

def find_path_cells(maze):
    cells = []
    height, width = len(maze), len(maze[0])
    offset_x = -MAZE_SIZE
    offset_y = -MAZE_SIZE
    
    for y in range(height):
        for x in range(width):
            if not maze[y][x]:
                wx = offset_x + x * GRID_SIZE + GRID_SIZE//2
                wy = offset_y + y * GRID_SIZE + GRID_SIZE//2
                cells.append((wx, wy))
    
    return cells

                           
              
                           
def distance_2d(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def get_gun_position(char_x, char_y, char_angle):
                                                                   
                                                                
    gun_offset_x = 16              
    gun_offset_y = 12                        
    gun_offset_z = 40                  
    
                                            
    rad = math.radians(char_angle)
                                                   
    world_x = char_x + gun_offset_x * math.sin(rad) + gun_offset_y * math.cos(rad)
    world_y = char_y + gun_offset_x * math.cos(rad) - gun_offset_y * math.sin(rad)
    world_z = gun_offset_z
    
    return [world_x, world_y, world_z]

def normalize_angle(angle):
    while angle >= 360: angle -= 360
    while angle < 0: angle += 360
    return angle

def check_wall_collision(x, y, radius):
    if game_state == GAME_OUTSIDE:
                            
        if abs(x) > OUTSIDE_SIZE - radius or abs(y) > OUTSIDE_SIZE - radius:
            return True
        
                                        
        hx, hy, hz = HOUSE_POS
        if abs(x - hx) < HOUSE_SIZE//2 + radius and abs(y - hy) < HOUSE_SIZE//2 + radius:
                              
            door_x, door_y = hx, hy - HOUSE_SIZE//2
            if abs(x - door_x) < 55 and abs(y - door_y) < 60:
                return False                  
            return True
        
        return False
    
    elif game_state == GAME_INSIDE:
                    
        for wall in walls:
            x1, y1, x2, y2, h = wall
            if point_to_segment_distance(x, y, x1, y1, x2, y2) < radius:
                return True
        
                  
        if abs(x) > MAZE_SIZE - radius or abs(y) > MAZE_SIZE - radius:
            return True
        
        return False
    
    return False

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) < 0.001 and abs(dy) < 0.001:
        return distance_2d(px, py, x1, y1)
    
    t = max(0, min(1, ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return distance_2d(px, py, proj_x, proj_y)

def line_intersects_wall(x1, y1, x2, y2):
    if game_state != GAME_INSIDE:
        return False
    
    for wall in walls:
        wx1, wy1, wx2, wy2, _ = wall
        if segments_intersect(x1, y1, x2, y2, wx1, wy1, wx2, wy2):
            return True
    return False

def segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 0.001:
        return False
    
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
    
    return (0 <= t <= 1) and (0 <= u <= 1)

def camera_has_clearance(cx, cy, walls, clearance):
    for wall in walls:
        x1, y1, x2, y2, h = wall
        dist = point_to_segment_distance(cx, cy, x1, y1, x2, y2)
        if dist < clearance:
            return False
    return True

                           
                     
                           
def init_outside():
    global trees, fence_posts, front_door_trigger, player_pos, player_angle
    
    player_pos = [0.0, -OUTSIDE_SIZE + 80, 0.0]
    player_angle = 90.0
    
                  
    trees = []
    for _ in range(20):                     
        tx = random.uniform(-OUTSIDE_SIZE + 100, OUTSIDE_SIZE - 100)
        ty = random.uniform(-OUTSIDE_SIZE + 100, OUTSIDE_SIZE - 100)
        
                          
        hx, hy, _ = HOUSE_POS
        if abs(tx - hx) > HOUSE_SIZE and abs(ty - hy) > HOUSE_SIZE:
            trees.append([tx, ty, random.uniform(0.8, 1.3)])
    
                                  
    fence_posts = []
    spacing = 80
    for i in range(-OUTSIDE_SIZE, OUTSIDE_SIZE, spacing):
        fence_posts.append([i, -OUTSIDE_SIZE, 1.0])
        fence_posts.append([i, OUTSIDE_SIZE, 1.0])
        fence_posts.append([-OUTSIDE_SIZE, i, 1.0])
        fence_posts.append([OUTSIDE_SIZE, i, 1.0])
    
                        
    hx, hy, _ = HOUSE_POS
    front_door_trigger = [hx, hy - HOUSE_SIZE//2, 70]
    
                    
    global rocks
    rocks = []
    for _ in range(8):
        rx = random.uniform(-OUTSIDE_SIZE + 150, OUTSIDE_SIZE - 150)
        ry = random.uniform(-OUTSIDE_SIZE + 150, OUTSIDE_SIZE - 150)
                               
        if abs(rx - hx) > HOUSE_SIZE and abs(ry - hy) > HOUSE_SIZE:
            if ry > -OUTSIDE_SIZE + 200:                          
                size = random.uniform(8, 18)
                rocks.append([rx, ry, size])

def init_inside():
    global maze_grid, walls, enemies, pickups, exit_door_trigger, player_pos, player_angle, inside_static_list, exit_door_open
    
                                                                                     
    maze_grid = generate_maze(GRID_TILES, GRID_TILES)
    maze_grid = carve_main_path(maze_grid)                                    
    maze_grid = carve_entrance_room(maze_grid)                 
                                                                       
    maze_grid = widen_maze(maze_grid)
    
    walls = maze_to_walls(maze_grid)
    
                        
    walls.extend([
        [-MAZE_SIZE, -MAZE_SIZE, MAZE_SIZE, -MAZE_SIZE, 100],
        [MAZE_SIZE, -MAZE_SIZE, MAZE_SIZE, MAZE_SIZE, 100],
        [MAZE_SIZE, MAZE_SIZE, -MAZE_SIZE, MAZE_SIZE, 100],
        [-MAZE_SIZE, MAZE_SIZE, -MAZE_SIZE, -MAZE_SIZE, 100],
    ])
    
    path_cells = find_path_cells(maze_grid)
    if not path_cells:
        path_cells = [(0, 0)]
    
                                
    spawn_cells = [c for c in path_cells if c[1] < -MAZE_SIZE + 200]
    if spawn_cells:
        player_pos = [float(spawn_cells[0][0]), float(spawn_cells[0][1]), 0.0]
    else:
        player_pos = [float(path_cells[0][0]), float(path_cells[0][1]), 0.0]
    
    player_angle = 90.0
    
                                          
    enemies = []
    level_def = LEVEL_DEFS.get(current_level, LEVEL_DEFS[1])
    enemy_count = level_def['enemy_count']
    enemy_hp = level_def['enemy_hp']
    
    far_cells = [c for c in path_cells if distance_2d(c[0], c[1], player_pos[0], player_pos[1]) > 250]
    
    if far_cells:
        for _ in range(min(enemy_count, len(far_cells))):
            ex, ey = random.choice(far_cells)
            far_cells.remove((ex, ey))
            enemies.append({'x': ex, 'y': ey, 'hp': enemy_hp, 'alive': True,
                            'angle': random.uniform(0, 360), 'state': 0,
                            'target_x': ex, 'target_y': ey, 'attack_cd': 0,
                            'rot_dir': 1, 'alerted': False, 'alert_timer': 0.0,
                            'corridor': None})
    
    pickups = []
    powerup_types = ['knife', 'gun', 'invisibility', 'distraction']
    far_pickup_cells = [c for c in path_cells if distance_2d(c[0], c[1], player_pos[0], player_pos[1]) > 150]
    
    if len(far_pickup_cells) >= 4:
        pickup_cells = random.sample(far_pickup_cells, 4)
        for i, (px, py) in enumerate(pickup_cells):
            pickups.append([px, py, powerup_types[i], False])
    
    exit_cells = [c for c in path_cells if c[1] > MAZE_SIZE - 200]
    if exit_cells:
        ex, ey = random.choice(exit_cells)
        exit_door_trigger = [ex, ey, 50]
    else:
        exit_door_trigger = [path_cells[-1][0], path_cells[-1][1], 50]
                                               
    exit_door_open = False
    
                                                                                   
                                                                           
    inside_static_list = None

def reset_game():
    global game_state, player_hp, player_ammo, player_weapon_damage, player_fire_rate
    global detection_level, detection_meter, detected, game_won, game_lost, bullets, fire_cooldown, animation_time
    global camera_pos, current_level, exit_door_open, camera_mode
    global player_weapon_state, player_powerups, player_gun_ammo, player_lives
    global invisibility_timer, invisibility_active, distraction
    global player_knife_hits, player_barehands_hits, player_collision_damage_cooldown
    
    game_state = GAME_OUTSIDE
    player_hp = PLAYER_MAX_HP
    player_ammo = 30
    player_weapon_damage = 1
    player_fire_rate = 1.0
    camera_mode = 0
    detection_level = 0.0
    detection_meter = 0.0
    detected = False
    seen_frames = 0
    game_won = False
    game_lost = False
    bullets = []
    fire_cooldown = 0
    animation_time = 0.0
    current_level = 1
    exit_door_open = False
    
    player_weapon_state = 0
    player_powerups = {'knife': False, 'gun': False, 'invisibility': False, 'distraction': False}
    player_gun_ammo = 0
    player_lives = 13
    invisibility_timer = 0.0
    invisibility_active = False
    distraction = None
    player_knife_hits = {}
    player_barehands_hits = {}
    player_collision_damage_cooldown = 0
    
    init_outside()
    camera_pos = [THIRD_PERSON_CAM_POS[0], THIRD_PERSON_CAM_POS[1], THIRD_PERSON_CAM_POS[2]]

                           
            
                           
def update_player():
    global player_hp, game_state, game_lost, last_game_state, current_level, exit_door_open
    
                         
    if game_state == GAME_OUTSIDE and front_door_trigger:
        dx, dy, radius = front_door_trigger
        if distance_2d(player_pos[0], player_pos[1], dx, dy) < radius:
                         
            last_game_state = game_state
            game_state = GAME_INSIDE
            init_inside()
            return
    
    elif game_state == GAME_INSIDE and exit_door_trigger:
        ex, ey, radius = exit_door_trigger
        if distance_2d(player_pos[0], player_pos[1], ex, ey) < radius:
                                                                       
            if exit_door_open:
                                               
                if current_level < 3:
                    current_level += 1
                    init_inside()
                else:
                                                  
                    last_game_state = game_state
                    game_state = GAME_WIN
            return
    
                 
    if player_hp <= 0:
        last_game_state = game_state
        game_state = GAME_LOSE


def update_bullets():
    global bullets, enemies, player_lives, game_state, last_game_state
    kept = []
    for bullet in bullets:
        bx, by, bz, vx, vy, vz, frames = bullet

        bx += vx
        by += vy
        bz += vz
        frames -= 1

        if frames <= 0 or bz < 0:
            continue

                        
        if check_wall_collision(bx, by, 5):
            continue

        # Check if bullet hits player
        if game_state == GAME_INSIDE:
            player_dist = distance_2d(bx, by, player_pos[0], player_pos[1])
            if player_dist < 15:
                player_lives -= 1
                if player_lives <= 0:
                    last_game_state = game_state
                    game_state = GAME_LOSE
                continue

                                       
        hit = False
        if game_state == GAME_INSIDE:
            for enemy in enemies:
                                                                              
                try:
                    if isinstance(enemy, dict):
                        ex = enemy.get('x')
                        ey = enemy.get('y')
                        if ex is None or ey is None:
                            continue
                        if distance_2d(bx, by, ex, ey) < ENEMY_RADIUS + 4:
                                          
                            if 'hp' in enemy:
                                enemy['hp'] -= 1
                                if enemy['hp'] <= 0:
                                    enemy['alive'] = False
                            else:
                                enemy['alive'] = False
                            hit = True
                            break
                    else:
                        ex, ey = enemy[0], enemy[1]
                        if distance_2d(bx, by, ex, ey) < ENEMY_RADIUS + 4:
                            enemy[2] -= 1
                            if enemy[2] <= 0:
                                enemy[2] = 0
                            hit = True
                            break
                except Exception:
                    continue

        if not hit:
            kept.append([bx, by, bz, vx, vy, vz, frames])

    bullets = kept

def update_pickups():
    global player_gun_ammo, player_powerups, invisibility_timer, invisibility_active
    
    if game_state != GAME_INSIDE:
        return
    
    for pickup in pickups:
        if pickup[3]:
            continue
        
        px, py, ptype, _ = pickup
        if distance_2d(player_pos[0], player_pos[1], px, py) < 30:
            pickup[3] = True
            
            if ptype == 'knife':
                player_powerups['knife'] = True
            elif ptype == 'gun':
                player_powerups['gun'] = True
                player_gun_ammo = 3
            elif ptype == 'invisibility':
                player_powerups['invisibility'] = True
                invisibility_timer = 5.0
                invisibility_active = True
            elif ptype == 'distraction':
                player_powerups['distraction'] = True

                           
        
                           
def update_camera():
    global camera_pos
    
    if camera_mode == 1:
        camera_pos[0] = player_pos[0]
        camera_pos[1] = player_pos[1]
        camera_pos[2] = 80
        return
    
    rad = math.radians(player_angle)
    camera_pos[0] = player_pos[0] - math.cos(rad) * 250
    camera_pos[1] = player_pos[1] - math.sin(rad) * 250
    camera_pos[2] = 450

def setup_camera():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    near_plane = 0.5 if camera_mode == 1 else 0.1
    gluPerspective(60, W/H, near_plane, 2000)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    if camera_mode == 1:
        rad = math.radians(player_angle)
        look_x = camera_pos[0] + math.cos(rad) * 100
        look_y = camera_pos[1] + math.sin(rad) * 100
        look_z = camera_pos[2]
        gluLookAt(camera_pos[0], camera_pos[1], camera_pos[2],
                  look_x, look_y, look_z,
                  0, 0, 1)
    else:
        gluLookAt(camera_pos[0], camera_pos[1], camera_pos[2],
                  player_pos[0], player_pos[1], 35,
                  0, 0, 1)

                           
                        
                           
def draw_ground(size, tile_size=100):
    glBegin(GL_QUADS)
    for x in range(-size, size, tile_size):
        for y in range(-size, size, tile_size):
                             
            if ((x//tile_size) + (y//tile_size)) % 2 == 0:
                glColor3f(0.3, 0.6, 0.3)
            else:
                glColor3f(0.25, 0.55, 0.25)
            
                                                 
            glVertex3f(x, y, 0)
            glVertex3f(x + tile_size, y, 0)
            glVertex3f(x + tile_size, y + tile_size, 0)
            glVertex3f(x, y + tile_size, 0)
    glEnd()

def draw_exterior_path():
    hx, hy, _ = HOUSE_POS
    spawn_y = -OUTSIDE_SIZE + 80
    
                             
    path_width = 70
    path_color = [0.45, 0.35, 0.25]         
    
    glColor3f(path_color[0], path_color[1], path_color[2])
    glBegin(GL_QUADS)
                                         
    glVertex3f(-path_width//2, spawn_y, 0.5)
    glVertex3f(path_width//2, spawn_y, 0.5)
    glVertex3f(path_width//2, hy - HOUSE_SIZE//2 - 30, 0.5)
    glVertex3f(-path_width//2, hy - HOUSE_SIZE//2 - 30, 0.5)
    glEnd()

def draw_sky():
                                                                
    
    glPushMatrix()
    glTranslatef(player_pos[0], player_pos[1], 0)
    
                             
    glBegin(GL_QUADS)
    glColor3f(0.4, 0.6, 0.9)                 
    glVertex3f(-2000, -2000, 800)
    glVertex3f(2000, -2000, 800)
    glColor3f(0.7, 0.8, 1.0)                         
    glVertex3f(2000, 2000, 200)
    glVertex3f(-2000, 2000, 200)
    glEnd()
    
    glPopMatrix()
    
                                                                   

def draw_baseboards():
    glColor3f(0.15, 0.12, 0.10)
    baseboard_height = 8
    
                                                             
                                                    
    bounds = [
        [-MAZE_SIZE, -MAZE_SIZE, MAZE_SIZE, -MAZE_SIZE],
        [MAZE_SIZE, -MAZE_SIZE, MAZE_SIZE, MAZE_SIZE],
        [MAZE_SIZE, MAZE_SIZE, -MAZE_SIZE, MAZE_SIZE],
        [-MAZE_SIZE, MAZE_SIZE, -MAZE_SIZE, -MAZE_SIZE],
    ]
    
    for x1, y1, x2, y2 in bounds:
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length < 0.1:
            continue
        
                                
        nx, ny = -dy/length, dx/length
        offset = 2
        fx1, fy1 = x1 + nx * offset, y1 + ny * offset
        fx2, fy2 = x2 + nx * offset, y2 + ny * offset
        
        glBegin(GL_QUADS)
                                             
        glVertex3f(fx1, fy1, 0)
        glVertex3f(fx2, fy2, 0)
        glVertex3f(fx2, fy2, baseboard_height)
        glVertex3f(fx1, fy1, baseboard_height)
        glEnd()

def draw_ceiling_beams():
    glColor3f(0.25, 0.20, 0.18)
    beam_width = 12
    beam_thickness = 8
    ceiling_z = 120
    
                                        
    for i in range(-3, 4):
        beam_y = i * 100
        glPushMatrix()
        glTranslatef(0, beam_y, ceiling_z - beam_thickness//2)
        glScalef(MAZE_SIZE * 1.8, beam_width, beam_thickness)
        glutSolidCube(1)
        glPopMatrix()

def draw_doorway_frames():
    glColor3f(0.3, 0.25, 0.22)
    frame_width = 60
    frame_height = 80
    frame_thickness = 8
    
                                      
    doorway_positions = [
        [0, -150],
        [0, 0],
        [0, 180],
    ]
    
    for dx, dy in doorway_positions:
                   
        glPushMatrix()
        glTranslatef(dx - frame_width//2, dy, frame_height//2)
        glScalef(frame_thickness, frame_thickness, frame_height)
        glutSolidCube(1)
        glPopMatrix()
        
                    
        glPushMatrix()
        glTranslatef(dx + frame_width//2, dy, frame_height//2)
        glScalef(frame_thickness, frame_thickness, frame_height)
        glutSolidCube(1)
        glPopMatrix()
        
                  
        glPushMatrix()
        glTranslatef(dx, dy, frame_height)
        glScalef(frame_width + frame_thickness * 2, frame_thickness, frame_thickness)
        glutSolidCube(1)
        glPopMatrix()

def draw_wall_segment(x1, y1, x2, y2, height, thickness=MAZE_WALL_THICKNESS):
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx*dx + dy*dy)
    if length < 0.1:
            return                              
    
    px, py = -dy/length * thickness/2, dx/length * thickness/2
    
    corners = [
        [x1 + px, y1 + py, 0],
        [x2 + px, y2 + py, 0],
        [x2 - px, y2 - py, 0],
        [x1 - px, y1 - py, 0]
    ]
    
            
    glColor3f(0.4, 0.35, 0.3)
    glBegin(GL_QUADS)
                                         
    for c in corners:
        glVertex3f(c[0], c[1], c[2])
    glEnd()
    
         
    glBegin(GL_QUADS)
                                         
    for i in [3, 2, 1, 0]:
        glVertex3f(corners[i][0], corners[i][1], height)
    glEnd()
    
                                              
                                                                          
    brick_color = [0.45, 0.30, 0.24]                       
    mortar_color = [0.28, 0.25, 0.22]                 
    
    for i in range(4):
        next_i = (i + 1) % 4
        fx1, fy1 = corners[i][0], corners[i][1]
        fx2, fy2 = corners[next_i][0], corners[next_i][1]
        fdx, fdy = fx2 - fx1, fy2 - fy1
        fn_len = math.sqrt(fdx*fdx + fdy*fdy)
        
        if fn_len < 0.001:
            continue
        
        nx, ny = -fdy/fn_len, fdx/fn_len
        
                              
                                                                      
        if i == 0 or i == 2:
            face_shade = 1.0
        else:
            face_shade = 0.88
        glColor3f(brick_color[0] * face_shade, brick_color[1] * face_shade, brick_color[2] * face_shade)
        
        glBegin(GL_QUADS)
                                             
        glVertex3f(corners[i][0], corners[i][1], 0)
        glVertex3f(corners[next_i][0], corners[next_i][1], 0)
        glVertex3f(corners[next_i][0], corners[next_i][1], height)
        glVertex3f(corners[i][0], corners[i][1], height)
        glEnd()
        
                                                                  
        if i == 0 or i == 2:
                                                           
            glColor3f(mortar_color[0], mortar_color[1], mortar_color[2])
            glLineWidth(1.5)
            
            glBegin(GL_LINES)
                                                                      
            for z_line in range(24, int(height), 48):
                glVertex3f(fx1, fy1, z_line)
                glVertex3f(fx2, fy2, z_line)
            
                                                    
            num_vlines = max(1, int(fn_len / 160))
            for v in range(1, num_vlines + 1):
                t = v / (num_vlines + 1)
                vx = fx1 + fdx * t
                vy = fy1 + fdy * t
                glVertex3f(vx, vy, 0)
                glVertex3f(vx, vy, height)
            glEnd()
            
            glLineWidth(1.0)

def draw_tree(x, y, scale):
    glPushMatrix()
    glTranslatef(x, y, 0)
    
           
    glColor3f(0.4, 0.25, 0.15)
    glPushMatrix()
    glTranslatef(0, 0, 20 * scale)
    gluCylinder(quadric, 8*scale, 6*scale, 30*scale, 8, 2)
    glPopMatrix()
    
                     
    glColor3f(0.2, 0.5, 0.2)
    glPushMatrix()
    glTranslatef(0, 0, 50 * scale)
    gluSphere(quadric, 25 * scale, 10, 10)
    glPopMatrix()

                               
    glPopMatrix()

def draw_house():
    hx, hy, hz = HOUSE_POS
    
    glPushMatrix()
    glTranslatef(hx, hy, hz)
    
                     
    door_width = 60
    door_height = 80
    wall_thickness = WALL_THICKNESS
    
                                    
    glColor3f(0.3, 0.25, 0.28)
                             
    glPushMatrix()
    glTranslatef(-HOUSE_SIZE//2 + (HOUSE_SIZE//2 - door_width//2)//2, -HOUSE_SIZE//2, 60)
    glScalef(HOUSE_SIZE//2 - door_width//2, wall_thickness, 120)
    glutSolidCube(1)
    glPopMatrix()
    
                              
    glPushMatrix()
    glTranslatef(HOUSE_SIZE//2 - (HOUSE_SIZE//2 - door_width//2)//2, -HOUSE_SIZE//2, 60)
    glScalef(HOUSE_SIZE//2 - door_width//2, wall_thickness, 120)
    glutSolidCube(1)
    glPopMatrix()
    
                                    
    glPushMatrix()
    glTranslatef(0, -HOUSE_SIZE//2, door_height + (120 - door_height)//2)
    glScalef(door_width, wall_thickness, 120 - door_height)
    glutSolidCube(1)
    glPopMatrix()
    
                       
    glPushMatrix()
    glTranslatef(0, HOUSE_SIZE//2, 60)
    glScalef(HOUSE_SIZE, wall_thickness, 120)
    glutSolidCube(1)
    glPopMatrix()
    
                       
    glPushMatrix()
    glTranslatef(-HOUSE_SIZE//2, 0, 60)
    glScalef(wall_thickness, HOUSE_SIZE, 120)
    glutSolidCube(1)
    glPopMatrix()
    
                        
    glPushMatrix()
    glTranslatef(HOUSE_SIZE//2, 0, 60)
    glScalef(wall_thickness, HOUSE_SIZE, 120)
    glutSolidCube(1)
    glPopMatrix()
    
          
    glColor3f(0.2, 0.15, 0.18)
    glPushMatrix()
    glTranslatef(0, 0, 125)
    glScalef(HOUSE_SIZE * 1.1, HOUSE_SIZE * 1.1, 10)
    glutSolidCube(1)
    glPopMatrix()
    
                    
    glColor3f(0.35, 0.3, 0.32)
    glPushMatrix()
    glTranslatef(0, -HOUSE_SIZE//2 - 25, 2)
    glScalef(door_width + 20, 40, 4)
    glutSolidCube(1)
    glPopMatrix()
    
                             
    glColor3f(0.25, 0.2, 0.22)
                
    glPushMatrix()
    glTranslatef(-door_width//2 - 3, -HOUSE_SIZE//2 - 2, door_height//2)
    glScalef(6, 8, door_height)
    glutSolidCube(1)
    glPopMatrix()
                 
    glPushMatrix()
    glTranslatef(door_width//2 + 3, -HOUSE_SIZE//2 - 2, door_height//2)
    glScalef(6, 8, door_height)
    glutSolidCube(1)
    glPopMatrix()
               
    glPushMatrix()
    glTranslatef(0, -HOUSE_SIZE//2 - 2, door_height + 3)
    glScalef(door_width + 12, 8, 6)
    glutSolidCube(1)
    glPopMatrix()
    
                                                                            
    pulse = 1.0 + 0.3 * math.sin(animation_time * 0.15)
    glColor3f(0.9 * pulse, 0.3 * pulse, 0.3 * pulse)
    glPushMatrix()
    glTranslatef(0, -HOUSE_SIZE//2 - 8, door_height + 15)
    gluSphere(quadric, 8 * pulse, 12, 12)
    glPopMatrix()
                                                
    glColor3f(0.4 * pulse, 0.12 * pulse, 0.12 * pulse)
    glPushMatrix()
    glTranslatef(0, -HOUSE_SIZE//2 - 8, door_height + 15)
    gluSphere(quadric, 15 * pulse, 12, 12)
    glPopMatrix()

                                
    glPopMatrix()

def draw_gun_firstperson():
    if player_weapon_state != 2:
        return
    
    glPushMatrix()
    
    glTranslatef(camera_pos[0], camera_pos[1], camera_pos[2])
    glRotatef(-player_angle + 90, 0, 0, 1)
    
    glTranslatef(30, 60, -40)
    
    glColor3f(0.8, 0.7, 0.6)
    glPushMatrix()
    glTranslatef(-10, 0, 0)
    glScalef(8, 15, 8)
    glutSolidCube(1)
    glPopMatrix()
    
    glColor3f(0.7, 0.7, 0.7)
    glPushMatrix()
    glTranslatef(5, 10, 0)
    glScalef(12, 20, 6)
    glutSolidCube(1)
    glPopMatrix()
    
    glColor3f(0.2, 0.2, 0.2)
    glPushMatrix()
    glTranslatef(0, 0, -5)
    glScalef(3, 3, 3)
    glutSolidCube(1)
    glPopMatrix()
    
    glPopMatrix()

def draw_roblox_character(x, y, angle, color):
                                                                   
    glPushMatrix()
    glTranslatef(x, y, 0)
    glRotatef(-angle + 90, 0, 0, 1)

                      
    base_height = 32
    head_size = 18
    torso_w, torso_h, torso_d = 16, 24, 10
    arm_w, arm_h, arm_d = 5, 16, 6
    leg_w, leg_h, leg_d = 6, 18, 6

                          
    swing = 0
    if player_moved_this_frame and (abs(x - player_pos[0]) < 1 and abs(y - player_pos[1]) < 1):
        swing = math.sin(animation_time * 0.3) * 20

           
    glPushMatrix()
    glTranslatef(0, 0, base_height)
    glScalef(torso_w, torso_d, torso_h)
    glColor3f(color[0], color[1], color[2])
    glutSolidCube(1)
    glPopMatrix()

                                       
    glPushMatrix()
    glTranslatef(0, 0, base_height + torso_h/2 + head_size/2)
    glScalef(head_size, head_size, head_size)
    glColor3f(1.0, 0.9, 0.8)
    glutSolidCube(1)
                                 
    glPushMatrix()
    glTranslatef(-head_size*0.18, head_size*0.28, 0.1)
    glScalef(0.18, 0.08, 0.08)
    glColor3f(1,1,1)
    glutSolidCube(1)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(head_size*0.18, head_size*0.28, 0.1)
    glScalef(0.18, 0.08, 0.08)
    glColor3f(1,1,1)
    glutSolidCube(1)
    glPopMatrix()
                               
    glPushMatrix()
    glTranslatef(-head_size*0.18, head_size*0.32, 0.2)
    glScalef(0.08, 0.04, 0.04)
    glColor3f(0,0,0)
    glutSolidCube(1)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(head_size*0.18, head_size*0.32, 0.2)
    glScalef(0.08, 0.04, 0.04)
    glColor3f(0,0,0)
    glutSolidCube(1)
    glPopMatrix()
    glPopMatrix()

                   
    for side in [-1, 1]:
        glPushMatrix()
        glTranslatef(side * (torso_w/2 + arm_w/2 + 1), 0, base_height + torso_h/2 - 2)
        glRotatef(swing * side, 1, 0, 0)
        glScalef(arm_w, arm_d, arm_h)
        glColor3f(0.9*color[0], 0.9*color[1], 0.9*color[2])
        glutSolidCube(1)
        glPopMatrix()

          
    for side in [-1, 1]:
        glPushMatrix()
        glTranslatef(side * (torso_w/4), 0, leg_h/2)
        glRotatef(-swing * side * 0.5, 1, 0, 0)
        glScalef(leg_w, leg_d, leg_h)
        glColor3f(0.15, 0.15, 0.15)
        glutSolidCube(1)
        glPopMatrix()

                                            
                                                           
    gun_x = torso_w/2 + 8
    gun_y = 0
    gun_z = base_height + torso_h/2
    
                                                        
    glPushMatrix()
    glTranslatef(gun_x, gun_y + 4, gun_z)
    glScalef(4, 10, 4)                
    glColor3f(0.2, 0.2, 0.2)                   
    glutSolidCube(1)
    glPopMatrix()
    
                             
    glPushMatrix()
    glTranslatef(gun_x, gun_y, gun_z - 6)
    glScalef(5, 4, 8)              
    glColor3f(0.3, 0.2, 0.15)              
    glutSolidCube(1)
    glPopMatrix()
    
                                  
    glPushMatrix()
    glTranslatef(gun_x, gun_y + 4, gun_z + 2)
    glScalef(3, 8, 3)
    glColor3f(0.25, 0.25, 0.25)              
    glutSolidCube(1)
    glPopMatrix()
    
                                      
    glPushMatrix()
    glTranslatef(gun_x, gun_y + 12, gun_z)
    glScalef(3, 2, 3)
    glColor3f(0.1, 0.1, 0.1)             
    glutSolidCube(1)
    glPopMatrix()

    glPopMatrix()

def draw_enemy(enemy):
    if isinstance(enemy, dict):
        ex = enemy.get('x', 0)
        ey = enemy.get('y', 0)
        hp = enemy.get('hp', 0)
        angle = enemy.get('angle', 0)
        alerted = enemy.get('alerted', False)
    else:
        ex, ey, hp, angle, state, tx, ty, cd = enemy

    if hp <= 0:
        return
    
    body_color = [0.45, 0.12, 0.12] if alerted else [0.25, 0.25, 0.25]
    draw_roblox_character(ex, ey, angle, body_color)
    
    eye_color = (1.0, 0.2, 0.2) if alerted else (1.0, 0.9, 0.2)
    head_size = 18
    torso_h = 24
    base_height = 32
    eye_z = base_height + torso_h/2 + head_size*0.2
    glPushMatrix()
    glTranslatef(ex, ey, eye_z)
    glRotatef(-angle + 90, 0, 0, 1)
    for side in [-1, 1]:
        glPushMatrix()
        glTranslatef(side * (head_size*0.18), head_size*0.28, 0.1)
        glColor3f(*eye_color)
        gluSphere(quadric, 1.6, 8, 8)
        glPopMatrix()
    glPopMatrix()
    
    glPushMatrix()
    glTranslatef(ex, ey, 10)
    glRotatef(-angle + 90, 0, 0, 1)
    
    glColor4f(1.0, 1.0, 0.0, 0.3) if not alerted else glColor4f(1.0, 0.0, 0.0, 0.5)
    glBegin(GL_TRIANGLES)
    rad_left = math.radians(-CONE_ANGLE)
    rad_right = math.radians(CONE_ANGLE)
    glVertex3f(0, 0, 0)
    glVertex3f(CONE_LENGTH * math.cos(rad_left), CONE_LENGTH * math.sin(rad_left), 0)
    glVertex3f(CONE_LENGTH * math.cos(rad_right), CONE_LENGTH * math.sin(rad_right), 0)
    glEnd()
    
    glPopMatrix()

def draw_pickup(pickup):
    px, py, ptype, collected = pickup
    
    if collected:
        return
    
    glPushMatrix()
    glTranslatef(px, py, 15 + math.sin(animation_time * 0.2) * 5)
    
    if ptype == 'knife':
        glColor3f(0.7, 0.7, 0.7)
    elif ptype == 'gun':
        glColor3f(0.3, 0.3, 0.3)
    elif ptype == 'invisibility':
        glColor3f(0.5, 0.8, 1.0)
    elif ptype == 'distraction':
        glColor3f(1.0, 0.5, 0.0)
    else:
        glColor3f(1.0, 1.0, 0.3)
    
    glutSolidCube(12)
    
    glPopMatrix()

def draw_bullet(bullet):
    bx, by, bz = bullet[:3]
    
    glPushMatrix()
    glTranslatef(bx, by, bz)
    
    glColor3f(1.0, 0.9, 0.3)
    gluSphere(quadric, 4, 8, 8)
    
    glPopMatrix()

def draw_floor_inside():
    tile_size = 120                                            
    base_color = [0.55, 0.45, 0.30]             
    alt_color = [0.50, 0.40, 0.27]                    
    
    glBegin(GL_QUADS)
                                         
    
    y = -MAZE_SIZE
    while y < MAZE_SIZE:
        x = -MAZE_SIZE
        while x < MAZE_SIZE:
                               
            if ((int(x / tile_size) + int(y / tile_size)) % 2 == 0):
                glColor3f(base_color[0], base_color[1], base_color[2])
            else:
                glColor3f(alt_color[0], alt_color[1], alt_color[2])
            
            glVertex3f(x, y, 0)
            glVertex3f(x + tile_size, y, 0)
            glVertex3f(x + tile_size, y + tile_size, 0)
            glVertex3f(x, y + tile_size, 0)
            
            x += tile_size
        y += tile_size
    
    glEnd()

def draw_ceiling_inside():
                                                                        
    return

def draw_entry_door_frame():
                                  
    frame_x = 0
    frame_y = -MAZE_SIZE + 80
    frame_width = 50
    frame_height = 70
    
    glColor3f(0.3, 0.25, 0.22)
    
               
    glPushMatrix()
    glTranslatef(frame_x - frame_width//2, frame_y, frame_height//2)
    glScalef(6, 8, frame_height)
    glutSolidCube(1)
    glPopMatrix()
    
                
    glPushMatrix()
    glTranslatef(frame_x + frame_width//2, frame_y, frame_height//2)
    glScalef(6, 8, frame_height)
    glutSolidCube(1)
    glPopMatrix()
    
              
    glPushMatrix()
    glTranslatef(frame_x, frame_y, frame_height)
    glScalef(frame_width + 12, 8, 6)
    glutSolidCube(1)
    glPopMatrix()

def draw_interior_props():
                             
    glColor3f(0.25, 0.2, 0.18)
    props = [
        [-200, -150, 15],
        [180, 220, 12],
        [-250, 180, 18],
        [150, -200, 14],
    ]
    for px, py, size in props:
        glPushMatrix()
        glTranslatef(px, py, size)
        glScalef(1.2, 0.8, 1.0)
        glutSolidCube(size * 2)
        glPopMatrix()
    
             
    glColor3f(0.2, 0.2, 0.23)
    pillars = [
        [0, 0],
        [200, 200],
        [-200, 200],
        [200, -200],
    ]
    for px, py in pillars:
        glPushMatrix()
        glTranslatef(px, py, 50)
        gluCylinder(quadric, 12, 12, 100, 8, 2)
        glPopMatrix()
    
                                                  
    lamp_positions = [
        [0, -250, 60],
        [0, -100, 60],
        [0, 100, 60],
        [0, 250, 60],
    ]
    for lx, ly, lz in lamp_positions:
        pulse = 0.7 + 0.2 * math.sin(animation_time * 0.1)
        glColor3f(0.6 * pulse, 0.5 * pulse, 0.3 * pulse)
        glPushMatrix()
        glTranslatef(lx, ly, lz)
        gluSphere(quadric, 6, 8, 8)
        glPopMatrix()
    

def draw_exit_door():
    if not exit_door_trigger:
        return
    
    ex, ey, radius = exit_door_trigger
    
    pulse = 1.0 + 0.3 * math.sin(animation_time * 0.2)
    
    if exit_door_open:
                                             
        glColor3f(0.2 * pulse, 1.0 * pulse, 0.2 * pulse)
    else:
                                    
        glColor3f(0.5, 0.2, 0.2)
    
    glPushMatrix()
    glTranslatef(ex, ey, 50)
    glutSolidCube(60)
    glPopMatrix()

                           
                
                           
def render_outside():
    glClearColor(0.5, 0.7, 0.9, 1.0)
    
    if camera_mode == 1:
        glPushMatrix()
        glTranslatef(camera_pos[0], camera_pos[1], camera_pos[2])
        
        glBegin(GL_QUADS)
        glColor3f(0.5, 0.7, 1.0)
        glVertex3f(-1500, -1500, 800)
        glVertex3f(1500, -1500, 800)
        glVertex3f(1500, 1500, 200)
        glVertex3f(-1500, 1500, 200)
        glEnd()
        
        glPopMatrix()
    else:
        draw_sky()
    
    draw_ground(OUTSIDE_SIZE)
    
                        
    draw_exterior_path()
    
           
    glColor3f(0.4, 0.4, 0.45)
    for rock in rocks:
        rx, ry, size = rock
        glPushMatrix()
        glTranslatef(rx, ry, size * 0.4)
        glScalef(1.2, 0.9, 0.7)                   
        glutSolidCube(size)
        glPopMatrix()
    
           
    for tree in trees:
        draw_tree(tree[0], tree[1], tree[2])
    
           
    glColor3f(0.4, 0.3, 0.25)
    for post in fence_posts:
        glPushMatrix()
        glTranslatef(post[0], post[1], 25)
        gluCylinder(quadric, 5, 5, 50, 8, 2)
        glPopMatrix()
    
    draw_house()
    
    if invisibility_active:
        glColor4f(0.3, 0.5, 0.9, 0.3)
    draw_roblox_character(player_pos[0], player_pos[1], player_angle, [0.3, 0.5, 0.9] if not invisibility_active else [0.5, 0.7, 1.0])
    
    if camera_mode == 1:
        draw_gun_firstperson()

def render_inside():
    glClearColor(0.1, 0.15, 0.2, 1.0)
    draw_floor_inside()
    draw_ceiling_inside()
    
                    
    for i, wall in enumerate(walls):
        variation = 0.02 if i % 2 == 0 else -0.02
        glColor3f(0.28 + variation, 0.28 + variation, 0.33 + variation)
        draw_wall_segment(wall[0], wall[1], wall[2], wall[3], wall[4])
    
             
    for pickup in pickups:
        draw_pickup(pickup)
    
               
    draw_exit_door()
    
    for enemy in enemies:
        draw_enemy(enemy)
    
    if invisibility_active:
        glColor4f(0.3, 0.5, 0.9, 0.3)
    draw_roblox_character(player_pos[0], player_pos[1], player_angle, [0.3, 0.5, 0.9] if not invisibility_active else [0.5, 0.7, 1.0])
    
    if camera_mode == 1:
        draw_gun_firstperson()
    
             
    for bullet in bullets:
        draw_bullet(bullet)

def draw_hud():
    glClear(GL_DEPTH_BUFFER_BIT)
    
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, W, 0, H)
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    glColor3f(1, 1, 1)
    
    draw_text(10, H - 30, f"Lives: {player_lives}/13")
    
    weapon_names = ['Bare Hands', 'Knife', 'Gun', 'Distraction']
    if player_weapon_state < len(weapon_names):
        weapon_text = weapon_names[player_weapon_state]
        if player_weapon_state == 2:
            weapon_text = f"Gun ({player_gun_ammo} bullets)"
        draw_text(10, H - 55, f"Weapon: {weapon_text}")
    
    if invisibility_active:
        glColor3f(0.5, 0.8, 1.0)
        draw_text(10, H - 80, f"Invisibility: {invisibility_timer:.1f}s")
    
    if game_state == GAME_INSIDE:
        glColor3f(0.5, 0.8, 1.0)
        draw_text(W - 150, H - 30, f"LEVEL: {current_level}/3")
        
        def enemy_alive(e):
            if isinstance(e, dict):
                return e.get('hp', 0) > 0 and e.get('alive', True)
            try:
                return e[2] > 0
            except Exception:
                return False
        living_enemies = [e for e in enemies if enemy_alive(e)]
        if exit_door_open:
            glColor3f(0.2, 1.0, 0.2)
            draw_text(W - 150, H - 55, "EXIT OPEN")
        else:
            glColor3f(1.0, 0.8, 0.2)
            draw_text(W - 150, H - 55, f"Enemies: {len(living_enemies)}")
    
    if paused:
        glColor3f(1, 1, 1)
        draw_text(W//2 - 40, H//2, "Paused")
    
    if game_state == GAME_OUTSIDE:
        glColor3f(1, 1, 0)
        draw_text(W//2 - 100, 50, "Find the Chamber!")
    elif game_state == GAME_WIN:
        glColor3f(0, 1, 0)
        draw_text(W//2 - 80, H//2, "YOU ESCAPED!")
        draw_text(W//2 - 80, H//2 - 30, "Press R to restart")
    elif game_state == GAME_LOSE:
        glColor3f(1, 0, 0)
        draw_text(W//2 - 100, H//2, "Game Over - Press R to Reset")
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_text(x, y, text):
    glRasterPos2f(x, y)
    for ch in text:
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(ch))

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glViewport(0, 0, W, H)
    
    setup_camera()
    
                                                                         
    if game_state == GAME_OUTSIDE:
        render_outside()
    elif game_state == GAME_INSIDE:
        render_inside()
    elif game_state == GAME_WIN or game_state == GAME_LOSE:
                                                                          
        if last_game_state == GAME_INSIDE:
            render_inside()
        else:
            render_outside()
    
    draw_hud()
    
    glutSwapBuffers()

                           
       
                           
def keyboard_down(key, x, y):
    global player_ammo, bullets, fire_cooldown, debug_mode, camera_mode
    global player_weapon_state, player_powerups, player_gun_ammo, player_lives, paused
    
                                                                       
    keys_pressed.add(key)
    key_timestamps[key] = int(time.perf_counter() * 1000)
    
    if key == b'r':
        reset_game()
        glutPostRedisplay()
    elif key == b'm':
        debug_mode = not debug_mode
    elif key == b'c' or key == b'C':
                                                                  
        camera_mode = 1 - camera_mode
        glutPostRedisplay()
    elif key == b'e' or key == b'E':
        if player_weapon_state == 0:
            if player_powerups.get('knife'):
                player_weapon_state = 1
            elif player_powerups.get('gun') and player_gun_ammo > 0:
                player_weapon_state = 2
            elif player_powerups.get('distraction'):
                player_weapon_state = 3
        elif player_weapon_state == 1:
            if player_powerups.get('gun') and player_gun_ammo > 0:
                player_weapon_state = 2
            elif player_powerups.get('distraction'):
                player_weapon_state = 3
            else:
                player_weapon_state = 0
        elif player_weapon_state == 2:
            if player_powerups.get('distraction'):
                player_weapon_state = 3
            else:
                player_weapon_state = 0
        elif player_weapon_state == 3:
            player_weapon_state = 0
        else:
            player_weapon_state = 0
        glutPostRedisplay()
    elif key == b'p' or key == b'P':
        paused = not paused
        glutPostRedisplay()
    elif key == b'f' or key == b' ':
        pass
    elif key == b'\x1b':       
        import sys
        sys.exit(0)

def keyboard_up(key, x, y):
                                                               
    keys_pressed.discard(key)
    if key in key_timestamps:
        del key_timestamps[key]

def special_key(key, x, y):
    global player_angle
    
    if key == GLUT_KEY_LEFT:
        player_angle = normalize_angle(player_angle + 3)
    elif key == GLUT_KEY_RIGHT:
        player_angle = normalize_angle(player_angle - 3)

def mouse(button, state, x, y):
    global player_ammo, bullets, fire_cooldown, player_weapon_state, player_gun_ammo, distraction, player_powerups

    if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
        if player_weapon_state == 2 and player_gun_ammo > 0 and fire_cooldown <= 0:
            player_gun_ammo -= 1
            if player_gun_ammo <= 0:
                player_powerups['gun'] = False
                player_weapon_state = 0
            fire_cooldown = int(FIRE_COOLDOWN / player_fire_rate)
            gun_pos = get_gun_position(player_pos[0], player_pos[1], player_angle)
            rad = math.radians(player_angle)
            vx = math.cos(rad) * BULLET_SPEED
            vy = math.sin(rad) * BULLET_SPEED
            bullets.append([gun_pos[0], gun_pos[1], gun_pos[2], vx, vy, 0, BULLET_LIFETIME])

    if button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN:
        if player_weapon_state == 3:
            rad = math.radians(player_angle)
            dx = math.cos(rad)
            dy = math.sin(rad)
            sx = player_pos[0] + dx * 10
            sy = player_pos[1] + dy * 10
            distraction = {'x': sx, 'y': sy, 'dx': dx, 'dy': dy, 'dist_left': 7.0, 'landed': False, 'created': time.perf_counter()}
            player_powerups['distraction'] = False
            player_weapon_state = 0

def update_enemies():
    global player_hp, game_state, last_game_state, invisibility_active, player_lives, bullets
    now = time.perf_counter()
    level_def = LEVEL_DEFS.get(current_level, LEVEL_DEFS[1])
    enemy_damage = level_def.get('enemy_damage', ENEMY_DAMAGE)
    
    for e in enemies:
        if not e.get('alive', True):
            continue
        
        old_angle = e.get('angle', 0)
        e['angle'] = (old_angle + e.get('rot_dir', 1) * 1.5) % 360
        
        rad = math.radians(e['angle'])
        cone_end_x = e['x'] + math.cos(rad) * CONE_LENGTH
        cone_end_y = e['y'] + math.sin(rad) * CONE_LENGTH
        
        if line_intersects_wall(e['x'], e['y'], cone_end_x, cone_end_y):
            e['rot_dir'] = -e.get('rot_dir', 1)
            e['angle'] = old_angle
        
        player_detectable = not invisibility_active
        
        dx = player_pos[0] - e['x']
        dy = player_pos[1] - e['y']
        dist = math.hypot(dx, dy)
        
        if dist <= CONE_LENGTH and player_detectable:
            ang_to_player = math.degrees(math.atan2(dy, dx))
            diff = abs(normalize_angle(ang_to_player - e['angle']))
            if diff > 180:
                diff = 360 - diff
            if diff < CONE_ANGLE and not line_intersects_wall(e['x'], e['y'], player_pos[0], player_pos[1]):
                e['alerted'] = True
                e['alert_timer'] = now
                
                if now > e.get('attack_cd', 0):
                    # Enemy shoots a bullet at player - aim at player's position
                    angle_to_player = math.atan2(dy, dx)
                    vx = math.cos(angle_to_player) * BULLET_SPEED * 0.7
                    vy = math.sin(angle_to_player) * BULLET_SPEED * 0.7
                    bullets.append([e['x'], e['y'], 20, vx, vy, 0, BULLET_LIFETIME])
                    e['attack_cd'] = now + 0.7
        
        if e.get('alerted'):
            if dist > 25 or not player_detectable:
                if now - e.get('alert_timer', now) > 2.0:
                    e['alerted'] = False
        
        if e.get('distraction_until', 0) > now:
            tx, ty = e.get('distraction_target', (e['x'], e['y']))
            e['angle'] = math.degrees(math.atan2(ty - e['y'], tx - e['x']))
        
        if e.get('alerted'):
            for other in enemies:
                if other is e or not other.get('alive'):
                    continue
                if other.get('corridor') == e.get('corridor'):
                    d = math.hypot(other['x'] - e['x'], other['y'] - e['y'])
                    if 10 <= d <= 15:
                        other['alerted'] = True
                        other['alert_timer'] = now


def update():
    global animation_time, player_moved_this_frame, fire_cooldown, last_update_time, player_angle
    global exit_door_open, player_lives, game_state, last_game_state
    
    current_time = int(time.perf_counter() * 1000)
    last_update_time = current_time

    for k, t in list(key_timestamps.items()):
        if current_time - t > KEY_HOLD_MS:
            keys_pressed.discard(k)
            del key_timestamps[k]

    if paused:
        glutPostRedisplay()
        return

    animation_time += 0.1
    player_moved_this_frame = False
    
    if fire_cooldown > 0:
        fire_cooldown -= 1
    
    if game_state in [GAME_WIN, GAME_LOSE]:
        glutPostRedisplay()
        return
    
                                                                        
    speed = MOVE_SPEED
    old_x, old_y = player_pos[0], player_pos[1]

                        
    if b'a' in keys_pressed:
        player_angle = normalize_angle(player_angle + 3)
    if b'd' in keys_pressed:
        player_angle = normalize_angle(player_angle - 3)

                  
    if b'w' in keys_pressed:
        rad = math.radians(player_angle)
        new_x = player_pos[0] + math.cos(rad) * speed
        new_y = player_pos[1] + math.sin(rad) * speed
        if not check_wall_collision(new_x, new_y, PLAYER_RADIUS):
            player_pos[0] = new_x
            player_pos[1] = new_y
    if b's' in keys_pressed:
        rad = math.radians(player_angle)
        new_x = player_pos[0] - math.cos(rad) * speed
        new_y = player_pos[1] - math.sin(rad) * speed
        if not check_wall_collision(new_x, new_y, PLAYER_RADIUS):
            player_pos[0] = new_x
            player_pos[1] = new_y
    
    if old_x != player_pos[0] or old_y != player_pos[1]:
        player_moved_this_frame = True
    
                       
    update_player()
    update_camera()
    update_bullets()
    update_pickups()
    update_enemies()
    global distraction, invisibility_timer, invisibility_active, player_collision_damage_cooldown, player_knife_hits, player_barehands_hits
    
    if player_collision_damage_cooldown > 0:
        player_collision_damage_cooldown -= 1
    
    if game_state == GAME_INSIDE and player_collision_damage_cooldown <= 0:
        for e in enemies:
            if not e.get('alive'):
                continue
            ex = e.get('x', 0)
            ey = e.get('y', 0)
            dist = distance_2d(player_pos[0], player_pos[1], ex, ey)
            
            if dist < PLAYER_RADIUS + ENEMY_RADIUS:
                eid = id(e)
                if player_weapon_state == 1:
                    player_knife_hits[eid] = player_knife_hits.get(eid, 0) + 1
                    if player_knife_hits[eid] >= 2:
                        e['hp'] = 0
                        e['alive'] = False
                    player_collision_damage_cooldown = 15
                elif player_weapon_state == 0:
                    player_barehands_hits[eid] = player_barehands_hits.get(eid, 0) + 1
                    if player_barehands_hits[eid] >= 3:
                        e['hp'] = 0
                        e['alive'] = False
                    player_collision_damage_cooldown = 15
    
    if distraction:
        if not distraction.get('landed'):
            step = 0.5
                                                         
            nx = distraction['x'] + distraction['dx'] * step
            ny = distraction['y'] + distraction['dy'] * step
                                  
            if line_intersects_wall(distraction['x'], distraction['y'], nx, ny):
                distraction['landed'] = True
            else:
                distraction['x'] = nx
                distraction['y'] = ny
                distraction['dist_left'] -= step
                if distraction['dist_left'] <= 0:
                    distraction['landed'] = True
        else:
                                                                 
            for e in enemies:
                if not e.get('alive'):
                    continue
                d = math.hypot(e['x'] - distraction['x'], e['y'] - distraction['y'])
                if d <= 5.0:
                    e['distraction_until'] = time.perf_counter() + 3.0
                    e['distraction_target'] = (distraction['x'], distraction['y'])
                                         
            if time.perf_counter() - distraction.get('created', 0) > 3.0:
                distraction = None

    if invisibility_timer > 0:
        invisibility_timer -= 0.1
        if invisibility_timer <= 0:
            player_powerups['invisibility'] = False
            invisibility_active = False
        else:
            invisibility_active = True
    else:
        invisibility_active = False

                                                                                        
                                                                           
    if game_state == GAME_INSIDE:
        living_enemies = [e for e in enemies if e.get('alive', True) and e.get('hp', 1) > 0]
        if not living_enemies:
            exit_door_open = True
    
    glutPostRedisplay()


def idle():
                                                                       
                                                                   
    current_ms = int(time.perf_counter() * 1000)
    if current_ms - last_update_time >= FRAME_TIME:
        update()

                           
      
                           
def main():
    global quadric
    
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(W, H)
    glutInitWindowPosition(100, 50)
    glutCreateWindow(b"3d Stealth Game")
    
                                                                 
                                                                                   
    quadric = gluNewQuadric()
                                                                                
    glEnable(GL_DEPTH_TEST)
    
    glClearColor(0.1, 0.15, 0.2, 1.0)
    
    reset_game()
    
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard_down)
                                                                                                 
    glutSpecialFunc(special_key)
    glutMouseFunc(mouse)
    glutIdleFunc(idle)                                                           
    
    glutMainLoop()

if __name__ == "__main__":
    main()
