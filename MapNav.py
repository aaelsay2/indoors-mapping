import os, glob
import numpy as np
import random
import math
import cv2
from MapFactory import MapFactory

class MapNav:
    def __init__(self, map_object, safe_offset=3, observation_dist=200, fov=60):
        self.map = map_object
        self.safe_offset = safe_offset
        self.observation_dist = observation_dist
        self.fov = fov
        self.init_config = None
        self.reset()
        self.initialize_config()

    def reset(self):
        self.not_mapped = self.map.get_inprogress_ids()
        self.collision = False
        self.score = 0
        self.cur_config = self.init_config
        self.sim_images = []

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

    def set_init_config(self, config):
        self.init_config = config
        self.cur_config = config

    def get_inflated(self):
        safe_offset = self.safe_offset
        inflated = self.map.clone()
        boundary = inflated.get_boundary()
        for el in inflated.get_elements():
            bbox = el.get_bbox()
            min_x = max(bbox['min'][0] - safe_offset, 0)
            min_y = max(bbox['min'][1] - safe_offset, 0)
            max_x = min(bbox['max'][0] + safe_offset, boundary[0])
            max_y = min(bbox['max'][1] + safe_offset, boundary[1])
            geom = [(min_x,min_y),(max_x,min_y),(max_x,max_y),(min_x,max_y)]
            el.set_geometry(geom)
            el.set_inprogress(False)
        return inflated

    def get_map(self):
        return self.map

    def check_collision(self, config):
        boundary = self.map.get_boundary()
        if config.x >= boundary[0] or config.x <= 0 or config.y >= boundary[1] or config.y <= 0:
            self.collision = True
            return True
        inflated = self.get_inflated()
        for el in inflated.get_elements():
            box = el.get_bbox()
            x_min = box['min'][0]
            y_min = box['min'][1]
            x_max = box['max'][0]
            y_max = box['max'][1]
            if config.x >= x_min and config.x <= x_max and config.y >= y_min and config.y <= y_max:
                self.collision = True
                return True
        self.collision = False
        return False

    def set_mapped(self,element_id):
        self.not_mapped.remove(element_id)


    def update_observation(self, config):
        observation_dist = self.observation_dist
        fov = self.fov
        observed = []
        for el in self.map.get_elements():
            if not(el.is_inprogress()):
                continue
            distance = self.get_distance(config, el)
            angle = self.get_angle(config, el)
            if distance<observation_dist and angle<(fov/2):
                elid = el.get_id()
                if elid in self.not_mapped:
                    observed.append(elid)
                    self.set_mapped(elid)
        return observed

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
        return 
        
    def initialize_config(self):
        boundary = self.map.get_boundary()
        map_range = boundary - np.ones(boundary.shape,dtype=int)*2*self.safe_offset
        for i in range(1000):
            x_rand = int(random.random()*map_range[0]+self.safe_offset)
            y_rand = int(random.random()*map_range[1]+self.safe_offset)
            theta_rand = random.randint(0,7)*45
            config = Config(x_rand, y_rand, theta_rand)
            collision = self.check_collision(config)
            if not(collision):
                self.set_init_config(config)
                return True
        raise ValueError('Can not generate a random init, try to create more empty spaces')
        return False

    def apply_action(self, action, render=True):
        # actions can be:
            # 0: do nothing
            # 1: move forward
            # 2: rotate CW by 45 deg
            # 3: rotate CCW by 45 deg
        if action==0:
            pass
        elif action==1:
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
        collision = self.check_collision(self.cur_config)

        # update observation
        observed = self.update_observation(self.cur_config)
        number_mapped = len(observed)
        self.score = self.score + number_mapped
        if number_mapped>0:
            print 'Mapped new elemenets', observed

        if render:
            self.render()


    def render(self):
        img = self.map.get_img()
        
        y_bound = self.map.get_boundary()[1]

        loc = (self.cur_config.x,self.cur_config.y)
        center = (loc[0], y_bound-loc[1])
        cv2.circle(img,center, 10, (0), -1)
        cv2.circle(img,center, self.safe_offset, (0), 1)
        cv2.circle(img,center, self.observation_dist, (0), 1)


        head = self.cur_config.theta
        head_pt = self.get_head_pt(loc, head, self.safe_offset)
        head_pt = (head_pt[0],y_bound-head_pt[1])
        cv2.line(img, center, head_pt, (0), 2)

        st_fov_head = int((head-self.fov/2+360)%360)
        st_fov_pt = self.get_head_pt(loc, st_fov_head, self.observation_dist)
        st_fov_pt = (st_fov_pt[0],y_bound-st_fov_pt[1])
        cv2.line(img, center, st_fov_pt, (0), 1)

        end_fov_head = int((head+self.fov/2)%360)
        end_fov_pt = self.get_head_pt(loc, end_fov_head, self.observation_dist)
        end_fov_pt = (end_fov_pt[0],y_bound-end_fov_pt[1])
        cv2.line(img, center, end_fov_pt, (0), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'Score: ' + str(self.score)
        cv2.putText(img,text ,(50,50), font, 0.7, (0), 2)

        text = 'Elements to map: ' + str(len(self.not_mapped))
        cv2.putText(img,text ,(50,80), font, 0.7, (0), 2)

        text = 'Collsion: ' + str(self.collision)
        cv2.putText(img,text ,(50,110), font, 0.8, (0), 2)

        self.sim_images.append(img)


        cv2.imshow('map',img)
        cv2.waitKey(5)
        # cv2.destroyAllWindows()

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
        

if __name__ == "__main__":
    factory = MapFactory()
    site_map = factory.create_map(1)
    nav = MapNav(site_map,30,200,90)
    conf = Config(400,100,0)
    nav.set_init_config(conf)

    nav.start_recording()

    for i in range(100):
        nav.apply_action(1)
    
    nav.apply_action(2)
    nav.apply_action(2)

    for i in range(300):
        nav.apply_action(1)

    nav.apply_action(3)

    for i in range(50):
        nav.apply_action(1)
    
    nav.apply_action(3)
    
    for i in range(500):
        nav.apply_action(1)

    nav.apply_action(2)
    nav.apply_action(2)

    for i in range(400):
        nav.apply_action(1)

    nav.end_recording()

# Convert images to video
# process = 'ffmpeg -framerate 30 -i simulation/sim_%d.jpg -c:v libx264 sim.avi'

