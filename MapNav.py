import os, glob
import numpy as np
import random
import math
import cv2
from MapFactory import MapFactory

class MapNav:
    def __init__(self, map_id, safe_offset=3, sensor_range=10, fov=60, resolution=50):
        self.map = MapFactory().create_map(map_id)
        self.safe_offset = safe_offset
        self.sensor_range = sensor_range
        self.fov = fov
        self.resolution = resolution
        self.init_config = None
        self.cur_config = None
        self.not_mapped = []
        self.incollision = False
        self.score = 0
        self.sim_images = []
        self.initialize_config()
        self.reset()


    def reset(self):
        self.not_mapped = self.map.get_inprogress()
        self.incollision = False
        self.score = 0
        self.set_init_config(self.init_config)
        self.sim_images = []
        self.hide_obstacles()

    def start_recording(self):
        self.sim_images = []

    def end_recording(self):
        try:  
            os.mkdir('./simulation')
        except OSError:
            files = glob.glob('./simulation/*')
            for f in files:
                os.remove(f)

        for (idx,img) in zip(range(len(self.sim_images)),self.sim_images):
            cv2.imwrite('./simulation/sim_'+str(idx)+'.jpg',img*255)
        
    def initialize_config(self):
        boundary = self.map.get_boundary()
        map_range = boundary - np.ones(boundary.shape,dtype=int)*2*self.safe_offset
        for i in range(1000):
            x_rand = int(random.random()*map_range[0]+self.safe_offset)
            y_rand = int(random.random()*map_range[1]+self.safe_offset)
            theta_rand = random.randint(0,7)*45
            config = Config(x_rand, y_rand, theta_rand)
            incollision = self.check_collision(config)
            if not(incollision):
                self.set_init_config(config)
                return True
        raise ValueError('Can not sample a random config in empty space, try to create more empty spaces')
        return False

    def set_init_config(self, config):
        self.init_config = config.clone()
        self.cur_config = config.clone()

    def inflate(self,element, safe_offset):
        boundary = self.map.get_boundary()
        bbox = element.get_bbox()
        min_x = max(bbox['min'][0] - safe_offset, 0)
        min_y = max(bbox['min'][1] - safe_offset, 0)
        max_x = min(bbox['max'][0] + safe_offset, boundary[0])
        max_y = min(bbox['max'][1] + safe_offset, boundary[1])
        geom = [(min_x,min_y),(max_x,min_y),(max_x,max_y),(min_x,max_y)]
        inflated = element.clone()
        inflated.set_geometry(geom)
        inflated.set_inprogress(False)
        return inflated

    def get_map(self):
        return self.map

    def detect_obstacles(self, config):
        for o in self.map.get_obstacles():
            distance = self.get_distance(config, o)
            if distance<=self.sensor_range:
                o.set_detected(True)

    def hide_obstacles(self):
        for o in self.map.get_obstacles():
            o.set_detected(False)

    def check_collision(self, config):
        boundary = self.map.get_boundary()
        if config.x >= boundary[0] or config.x <= 0 or config.y >= boundary[1] or config.y <= 0:
            self.incollision = True
            return True
        objects = self.map.get_elements()+self.map.get_obstacles()
        collision = False
        for o in objects:
            distance = self.get_distance(config, o)
            if distance<=self.safe_offset:
                collision = True
                break
        self.incollision = collision
        return collision

    def set_mapped(self, element):
        indices = [i for i,el in enumerate(self.not_mapped) if element.get_id() == el.get_id()]
        if len(indices)>0:
            del self.not_mapped[indices[0]]
            return True
        return False

    def update_observation(self, config):
        sensor_range = self.sensor_range
        fov = self.fov
        observed = []
        for el in self.map.get_elements():
            if not(el.is_inprogress()):
                continue
            distance = self.get_distance(config, el)
            angle = self.get_angle(config, el)
            if distance<=sensor_range and angle<=(fov/2):
                mapped = self.set_mapped(el)
                if mapped:
                    observed.append(el)
        return observed

    def backproject(self, config, elements):
        pass

    def elements_in_range(self, config):
        sensor_range = self.sensor_range
        fov = self.fov
        observed = []
        # First detect elements inside the FOV and range to reduce the backprojection computation
        for el in self.map.get_elements():
            distance = self.get_distance(config, el)
            angle = self.get_angle(config, el)
            if distance<sensor_range and angle<(fov/2):
                observed.append(el)

        #backproject the elements to checj visibility
        zbuffer = np.ones([1,self.resolution])

    def get_distance(self, config, element):
        box = element.get_bbox()
        x_min = box['min'][0]
        y_min = box['min'][1]
        x_max = box['max'][0]
        y_max = box['max'][1]

        x_clamp = min(max(config.x,x_min),x_max)
        y_clamp = min(max(config.y,y_min),y_max)
        
        if config.x>x_min and config.x<x_max:
            x_clamp = config.x
        elif config.y>y_min and config.y<y_max:
            y_clamp = config.y        
        
        distance = pow(pow(config.x-x_clamp,2)+pow(config.y-y_clamp,2),0.5)
        return distance

    def get_angle(self, config, element):
        box = element.get_bbox()
        x_p = (box['min'][0] + box['max'][0])/2
        y_p = (box['min'][1] + box['max'][1])/2
        heading = math.atan2(x_p-config.x, y_p-config.y)/math.pi*180
        delta = abs(heading-config.theta)
        return delta

    def get_input_size(self):
        return self.map.get_boundary()

    def get_possible_actions(self):
        return [1,2,3]

    def step(self, action_idx):
        action = self.get_possible_actions()[action_idx]
        reward = self.apply_action(action, False)
        done = len(self.not_mapped) == 0
        collision = self.incollision
        return reward, collision, done

    def apply_action(self, action, render=True):
        # actions can be:
            # 1: move forward
            # 2: rotate CW by 45 deg
            # 3: rotate CCW by 45 deg
        if action==1:
            heading = math.pi*self.cur_config.theta/180
            x_add = int(round(math.sin(heading)))
            y_add = int(round(math.cos(heading)))
            self.cur_config.x = self.cur_config.x + x_add
            self.cur_config.y = self.cur_config.y + y_add
        elif action==2:
            new_theta = self.cur_config.theta+45
            self.cur_config.theta = new_theta % 360
        elif action==3:
            new_theta = self.cur_config.theta-45+360
            self.cur_config.theta = new_theta % 360

        # update collision
        self.detect_obstacles(self.cur_config)
        incollision = self.check_collision(self.cur_config)

        # update observation
        observed = self.update_observation(self.cur_config)
        number_mapped = len(observed)
        self.score = self.score + number_mapped
        # if number_mapped>0:
        #     print 'Mapped new elemenets', observed

        if render:
            self.render()

        return number_mapped


    def render(self, render_scale=10, complete=True, store=True):
        img = self.map.get_img(render_scale=render_scale, complete=False)
        # Render the agent
        y_bound = self.map.get_boundary()[1]*render_scale
        safe_offset = self.safe_offset*render_scale
        sensor_range = self.sensor_range*render_scale

        loc = (self.cur_config.x*render_scale,self.cur_config.y*render_scale)
        center = (loc[0], y_bound-loc[1])
        cv2.circle(img, center, safe_offset, (0.25), -1)
        if complete:
            cv2.circle(img,center, safe_offset, (0), 1)
            cv2.circle(img,center, sensor_range, (0), 1)

        head = self.cur_config.theta
        head_pt = self.get_head_pt(loc, head, safe_offset)
        head_pt = (head_pt[0],y_bound-head_pt[1])
        cv2.line(img, center, head_pt, (0), 2)

        if complete:
            st_fov_head = int((head-self.fov/2+360)%360)
            st_fov_pt = self.get_head_pt(loc, st_fov_head, sensor_range)
            st_fov_pt = (st_fov_pt[0],y_bound-st_fov_pt[1])
            cv2.line(img, center, st_fov_pt, (0), 1)

            end_fov_head = int((head+self.fov/2)%360)
            end_fov_pt = self.get_head_pt(loc, end_fov_head, sensor_range)
            end_fov_pt = (end_fov_pt[0],y_bound-end_fov_pt[1])
            cv2.line(img, center, end_fov_pt, (0), 1)

        if complete:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'Score: ' + str(self.score)
            cv2.putText(img,text ,(50,50), font, 0.7, (0), 2)

            text = 'Elements to map: ' + str(len(self.not_mapped))
            cv2.putText(img,text ,(50,80), font, 0.7, (0), 2)

            text = 'Collsion: ' + str(self.incollision)
            cv2.putText(img,text ,(50,110), font, 0.8, (0), 2)

        if store:
            self.sim_images.append(img)
        # cv2.destroyAllWindows()
        return img

    def get_head_pt(self, start, head, distance):
        angle = float(head)/180.0*math.pi
        head_x = math.sin(angle)*distance+start[0]
        head_y = math.cos(angle)*distance+start[1]
        pt = (int(head_x),int(head_y))
        return pt


class Config:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def clone(self):
        config = Config(self.x, self.y, self.theta)
        return config

if __name__ == "__main__":

    nav = MapNav(map_id=1, safe_offset=3, sensor_range=20, fov=60, resolution=50)
    conf = Config(40,10,0)
    nav.set_init_config(conf)

    nav.start_recording()

    for i in range(6):
        nav.apply_action(1)
    
    nav.apply_action(2)
    nav.apply_action(2)

    for i in range(30):
        nav.apply_action(1)

    nav.apply_action(3)

    for i in range(5):
        nav.apply_action(1)
    
    nav.apply_action(3)
    
    for i in range(40):
        nav.apply_action(1)

    nav.apply_action(2)
    nav.apply_action(2)

    for i in range(40):
        nav.apply_action(1)

    nav.end_recording()

# Convert images to video
# process = 'ffmpeg -framerate 10 -i simulation/sim_%d.jpg -c:v libx264 sim.avi'

