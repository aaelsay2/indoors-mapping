import cv2
import matplotlib.pyplot as plt
import numpy as np

class MapFactory:
    def __init__(self):
        self.map = None

    def create_map(self, id):
        map = None
        if id==1:
            map = self.map1()
        elif id==2:
            pass
        elif id==3:
            pass
        else:
            pass
        return map

    def map1(self):
        map = Map()
        map.set_boundary([100,70])
        map.add_element(Element('column1', [(0,0),(2,0),(2,4),(0,4)],False))
        map.add_element(Element('column2', [(20,0),(24,0),(24,4),(20,4)],False))
        map.add_element(Element('column3', [(40,0),(44,0),(44,4),(40,4)],False))
        map.add_element(Element('column4', [(56,0),(60,0),(60,4),(56,4)],False))
        map.add_element(Element('column5', [(76,0),(80,0),(80,4),(76,4)],False))
        map.add_element(Element('column6', [(97,0),(99,0),(99,4),(97,4)],False))
        map.add_element(Element('column7', [(0,20),(2,20),(2,24),(0,24)],False))
        map.add_element(Element('column8', [(28,20),(32,20),(32,24),(28,24)],False))
        map.add_element(Element('column9', [(48,20),(52,20),(52,24),(48,24)],False))
        map.add_element(Element('column10', [(68,20),(72,20),(72,24),(68,24)],True))
        map.add_element(Element('column11', [(97,20),(99,20),(99,24),(97,24)],True))
        map.add_element(Element('wall12', [(0,44),(24,44),(24,46),(0,46)],False))
        map.add_element(Element('wall13', [(40,44),(49,44),(49,46),(40,46)],False))
        map.add_element(Element('wall14', [(51,44),(69,44),(69,46),(51,46)],True))
        map.add_element(Element('wall15', [(76,44),(99,44),(99,46),(76,46)],True))
        map.add_element(Element('wall16', [(49,44),(51,44),(51,69),(49,69)],False))
        map.add_element(Element('column17', [(28,60),(32,60),(32,64),(28,64)],False))
        map.add_element(Element('column18', [(68,60),(72,60),(72,64),(68,64)],True))
        map.add_obstacle(Obstacle([(50,16),(64,16),(64,18),(50,18)]))
        map.add_obstacle(Obstacle([(30,30),(34,30),(34,34),(30,34)]))
        map.add_obstacle(Obstacle([(69,24),(74,24),(74,30),(69,30)]))
        map.add_obstacle(Obstacle([(18,46),(20,46),(20,64),(18,64)]))
        map.add_obstacle(Obstacle([(58,50),(62,50),(62,64),(58,64)]))
        
        return map


class Map:
    def __init__(self):
        self.boundary = np.array([0,0])
        self.elements = []
        self.obstacles = []

    def clone(self):
        map = Map()
        map.set_boundary(self.boundary)
        for el in self.elements:
            new_el = Element(el.get_name(), el.get_geometry(), el.is_inprogress())
            map.add_element(new_el)

        for ob in self.obstacles:
            new_ob = Obstacle(ob.get_geometry())
            map.add_obstacle(new_ob)

        return map

    def get_boundary(self):
        return self.boundary

    def set_boundary(self,bound):
        if isinstance(bound, list) and len(bound) == 2:
            self.boundary = np.array(bound)
        else:
            raise TypeError('Boundary has to be a list of length 2')

    def add_element(self, element):
        new_index = 1
        if len(self.elements)>0:
            new_index = new_index + self.elements[-1].get_id()
        element.set_id(new_index)
        self.elements.append(element)
    
    def add_obstacle(self, obstacle):
        new_index = 1
        if len(self.obstacles)>0:
            new_index = new_index + self.obstacles[-1].get_id()
        obstacle.set_id(new_index)
        self.obstacles.append(obstacle)
    
    def get_elements(self):
        return self.elements

    def get_obstacles(self):
        return self.obstacles

    def get_inprogress(self):
        return [el for el in self.elements if el.is_inprogress()]

    def get_element_index(self, element):
        indices = [idx for i,el in enumerate(self.elements) if element.get_id() == el.get_id()]
        if len(indices)>0:
            return indices[0]
        return None

    def get_obstacle_index(self, obstacle):
        for ob,idx in zip(self.obstacles, range(len(self.obstacles))):
            if obstacle.get_id() == ob.get_id():
                return idx
        return None

    def get_img(self, scale=10):
        boundary = self.get_boundary()*scale-scale
        img = np.ones(np.flip(boundary))
        for el in self.elements:
            geom = el.get_geometry()
            shape = np.zeros(geom.shape,dtype=int)
            shape[:,0] = geom[:,0]*scale
            shape[:,1] = boundary[1]-geom[:,1]*scale
            progress = el.is_inprogress()
            if progress:
                cv2.fillPoly(img,pts=[shape],color=0.5)
            else:
                cv2.fillPoly(img,pts=[shape],color=0)

        for ob in self.obstacles:
            if ob.is_detected():
                geom = ob.get_geometry()
                shape = np.zeros(geom.shape,dtype=int)
                shape[:,0] = geom[:,0]*scale
                shape[:,1] = boundary[1]-geom[:,1]*scale
                cv2.fillPoly(img,pts=[shape],color=0)
                # cv2.polylines(img,pts=[shape],isClosed=True,color=0,thickness=3)

        for el in self.elements:
            elid = str(el.get_id())
            font = cv2.FONT_HERSHEY_SIMPLEX
            loc_max = (el.get_bbox()['max']*scale)
            loc_min = (el.get_bbox()['min']*scale)
            loc = [loc_max[0]+scale, boundary[1]-loc_max[1]-scale]
            if loc_max[0]>=boundary[0]-scale:
                loc[0] = loc_min[0]-scale
            if loc_max[1]>=boundary[1]-scale:
                loc[1] = boundary[1]-loc_min[1]-scale
            cv2.putText(img,elid,tuple(loc), font, 0.5,(0),1)
        return img

    def show_img(self):
        img = self.get_img(10)
        cv2.imshow('map',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_img(self, name='map'):
        img = self.get_img(10)
        cv2.imwrite(name+'.png',img*255)

class Obstacle:
    def __init__(self, geometry):
        self.id = None
        self.geometry = None
        self.detected = False
        self.set_geometry(geometry)
    
    def set_id(self, id):
        if self.id == None:
            self.id = id
        else:
            print('Can not reassign an id for obstacle with id %s'.format(self.id))

    def get_id(self):
        return self.id

    def get_geometry(self):
        return self.geometry

    def set_geometry(self, geometry):
        geom = np.array(geometry)
        self.geometry = geom
    
    def is_detected(self):
        return self.detected

    def set_detected(self, detected):
        self.detected = detected
        
    def get_bbox(self):
        geometry = self.get_geometry()
        minX = min(geometry[:,0])
        minY = min(geometry[:,1])
        maxX = max(geometry[:,0])
        maxY = max(geometry[:,1])
        return {'min':np.array([minX,minY]),'max':np.array([maxX,maxY])}

    def get_center(self):
        bbox = self.get_bbox()
        centerX = float(bbox['min'][0]+bbox['max'][0])/2
        centerY = float(bbox['min'][1]+bbox['max'][1])/2
        return [centerX,centerY]
    
    def set_inprogress(self, state):
        pass

    def clone(self):
        obstacle = Obstacle(self.geometry)
        return obstacle
        

class Element(Obstacle):
    def __init__(self, name, geometry, inprogress=False):   #Assuing the geometry is a bounding box for now
        self.id = None
        self.name = None
        self.geometry = None
        self.inprogress = None
        self.mapped = False
        self.set_name(name)
        self.set_geometry(geometry)
        self.set_inprogress(inprogress)

    def get_name(self):
        return self.name
        
    def set_name(self, name):
        self.name = name

    def is_inprogress(self):
        return self.inprogress

    def set_inprogress(self, state):
        self.inprogress = state

    def clone(self):
        element = Element(self.name, self.geometry,self.inprogress)
        return element

if __name__ == "__main__":
    factory = MapFactory()
    map = factory.create_map(1)
    map.show_img()
    map.save_img()