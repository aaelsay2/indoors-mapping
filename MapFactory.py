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
        map.set_boundary([1000,1000])
        map.add_element(Element('1', 'column', [(0,0),(20,0),(20,20),(0,20)]))
        map.add_element(Element('2', 'column', [(240,0),(260,0),(260,20),(240,20)]))
        map.add_element(Element('3', 'column', [(480,0),(520,0),(520,20),(480,20)]))
        map.add_element(Element('4', 'column', [(740,0),(760,0),(760,20),(740,20)]))
        map.add_element(Element('5', 'column', [(980,0),(999,0),(999,20),(980,20)]))        
        map.add_element(Element('6', 'column', [(240,240),(260,240),(260,260),(240,260)],True))
        map.add_element(Element('7', 'column', [(480,240),(520,240),(520,260),(480,260)],True))
        map.add_element(Element('8', 'column', [(740,240),(760,240),(760,260),(740,260)], True))
        map.add_element(Element('9_1', 'wall', [(0,500),(200,500),(200,520),(0,520)]))
        map.add_element(Element('9_2', 'wall', [(300,500),(500,500),(500,520),(300,520)]))
        map.add_element(Element('10_1', 'wall', [(500,500),(700,500),(700,520),(500,520)],True))
        map.add_element(Element('10_2', 'wall', [(800,500),(999,500),(999,520),(800,520)],True))
        map.add_element(Element('11', 'column', [(240,800),(260,800),(260,820),(240,820)]))
        map.add_element(Element('12', 'wall', [(490,520),(510,520),(510,999),(490,999)]))
        map.add_element(Element('13', 'column', [(740,800),(760,800),(760,820),(740,820)],True))
        map.add_element(Element('14', 'wall', [(0,960),(20,960),(20,999),(0,999)]))
        map.add_element(Element('15', 'wall', [(980,960),(999,960),(999,999),(980,999)],True))
        return map


class Map:
    def __init__(self):
        self.boundary = [1000,1000]
        self.elements = []

    def clone(self):
        map = Map()
        map.set_boundary(self.boundary)
        for el in self.elements:
            new_el = Element(el.get_id() , el.get_category(), el.get_geometry(), el.is_inprogress())
            map.add_element(new_el)
        return map

    def get_boundary(self):
        return np.array(self.boundary)

    def set_boundary(self,bound):
        if isinstance(bound, list) and len(bound) == 2:
            self.boundary = bound
        else:
            raise TypeError('Boundary has to be a list of length 2')

    def add_element(self, element):
        if isinstance(element, Element):
            el, idx = self.get_element(element.get_id())
            if idx == None:
                self.elements.append(element) 
        else:
            raise TypeError('Added element is not of type Element')

    def remove_element(self, id):
        el, idx = self.get_element(id)
        if not(idx == None):
            self.elements[idx] = []
        else:
            raise TypeError('Element to be removed does not exist or duplicated')

    def get_element(self,id):
        indices = [i for (e, i) in zip(self.elements, range(len(self.elements))) if e.get_id() == id]
        if len(indices) == 1:
            return self.elements[indices[0]], indices[0]
        else:
            return None,None
    
    def get_elements(self):
        return self.elements

    def get_inprogress_ids(self):
        return [el.get_id() for el in self.elements if el.is_inprogress()]

    def get_img(self):
        boundary = self.get_boundary()
        img = np.ones(boundary)
        for el in self.elements:
            geom = el.get_geometry()
            new_geom = np.zeros(geom.shape,dtype=int)
            new_geom[:,0] = geom[:,0]
            new_geom[:,1] = boundary[1]-geom[:,1]            
            progress = el.is_inprogress()
            if progress:
                cv2.fillPoly(img,pts=[new_geom],color=0.5)
            else:
                cv2.fillPoly(img,pts=[new_geom],color=0)
            
        for el in self.elements:
            elid = el.get_id()
            font = cv2.FONT_HERSHEY_SIMPLEX
            loc = el.get_bbox()['max']+5
            loc[1] = boundary[1]-loc[1]
            cv2.putText(img,elid,tuple(loc), font, 0.5,(0),2)

        return img

    def show_img(self):
        img = self.get_img()
        cv2.imshow('map',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_img(self, name='map'):
        img = self.get_img()
        cv2.imwrite(name+'.png',img*255)

class Element:
    def __init__(self, id, category, geometry, inprogress=False):
        self.id = id
        self.category = None
        self.geometry = None
        self.bbox = None
        self.inprogress = None
        self.set_category(category)
        self.set_geometry(geometry)
        self.set_inprogress(inprogress)
    
    def get_id(self):
        return self.id

    def get_category(self):
        return self.category
        
    def set_category(self, category):
        self.category = category

    def get_geometry(self):
        return self.geometry

    def set_geometry(self, geometry):
        geom = np.array(geometry)
        self.geometry = geom
        bbox = {'min': np.min(geom, axis=0), 'max':np.max(geom, axis=0)}
        self.bbox = bbox

    def get_bbox(self):
        return self.bbox

    def is_inprogress(self):
        return self.inprogress

    def set_inprogress(self, state):
        self.inprogress = state


if __name__ == "__main__":
    factory = MapFactory()
    map = factory.create_map(1)
    map.show_img()
    map.save_img()