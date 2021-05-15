# import 
from threading import Thread 
import multiprocessing
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from multiprocessing import Process
import os
import warnings
warnings.filterwarnings("ignore")

# ===== Region =====
class list_region:
    '''Lists where regions are stored'''
    def __init__(self, img, size_square_R):
        self.img = img
        self.size_square_R = size_square_R
        self.R = list()
        self.indices = list()
        self.i_j = list()
        self.dico_index_size = dict()
    
    def create_regions(self):
        k=0
        for i in range(0,self.img.shape[0],self.size_square_R):
            for j in range(0,self.img.shape[1],self.size_square_R):
                self.R.append(self.img[i:i+self.size_square_R,j:j+self.size_square_R])
                self.indices.append(k)
                self.i_j.append([i,j])
                self.dico_index_size[tuple([i,j])] = self.size_square_R
                k+=1  
        self.R = np.array(self.R)
        
# ===== Region =====
class list_domain():
    '''Lists where domains are stored'''
    def __init__(self, img, size_square_R):
        self.img = img
        self.size_square_R = size_square_R
        self.D = list()
        self.indices = list()
        self.i_j = list()
    
    def create_domains(self):
        k=0
        N = self.img.shape[1]
        for i in range(0,self.img.shape[0],(2*self.size_square_R)):
            for j in range(0,self.img.shape[1],(2*self.size_square_R)):
                self.i_j.append([i,j])
                self.D.append(self.img[i:i+2*self.size_square_R,j:j+2*self.size_square_R])
                self.indices.append([k,k+1,k+N,k+1+N])
                k+=1
        self.D = np.array(self.D)
    
    def create_domains_complete(self):
        self.D = self.D.tolist() 
        k=1
        N = self.img.shape[1]
        for i in range(self.size_square_R,self.img.shape[0]-2*self.size_square_R, self.size_square_R // 2):
            for j in range(self.size_square_R,self.img.shape[1]-2*self.size_square_R, self.size_square_R // 2):
                self.i_j.append([i,j])
                self.D.append(self.img[i:i+2*self.size_square_R,j:j+2*self.size_square_R])
                self.indices.append([k,k+1,k+N,k+1+N])
                k+=1
                
                
        self.D = np.array(self.D)
                 
# ===== Enhance methods =====
class enhance_img:
    def __init__(self, img, size_square_R=32, tol=6, size=512):
        self.size = size
        self.img = img
        self.img = self.get_greyscale_image()
        self.img = self.img[:self.size,:self.size]
        
        self.size_square_R = size_square_R
        self.tol = tol
        
        self.R_ = list_region(self.img,size_square_R)
        self.R_.create_regions()
        self.D_ = list_domain(self.img,size_square_R)
        self.D_.create_domains()
        #print(len(self.D_.D))
        #self.D_.create_domains_complete()
        #print(len(self.D_.D))
        
        self.dico_D = dict()
        self.dico_D[size_square_R] = self.D_
        
        self.g = None
        self.f1 = None
        self.f2 = None
        self.f3 = None
        
    def split(self, i, R, i_j, dico_index_size, indices_R):
        '''QuadTree spliting function : 
                input : i = indice of region to split
                            list : R = list of region
                            list : i_j = grid index of region on the list
                            dict : dico_index_size = dictionary of size for region key=>index value=>size
                            list : indices_R = new indices of regions i_j
                output :
                    updated input
        '''
        
        size_QuadTree = dico_index_size[tuple(i_j[i])]//2
        
        # new size of spliting region
        dico_index_size[tuple(i_j[i])] = size_QuadTree
        dico_index_size[tuple([i_j[i][0],i_j[i][1]+size_QuadTree])] = size_QuadTree
        dico_index_size[tuple([i_j[i][0]+size_QuadTree,i_j[i][1]])] = size_QuadTree
        dico_index_size[tuple([i_j[i][0]+size_QuadTree,i_j[i][1]+size_QuadTree])] = size_QuadTree
        
        if type(R[i]) == list:
            Rt = np.array(R[i])
        else:
            Rt = R[i]
            
        # split region
        reg = [Rt[0:size_QuadTree,0:size_QuadTree],Rt[0:size_QuadTree,size_QuadTree:2*size_QuadTree],Rt[size_QuadTree:2*size_QuadTree,0:size_QuadTree],Rt[size_QuadTree:2*size_QuadTree,size_QuadTree:2*size_QuadTree]]
        
        # update Region R 
        tmp = R.tolist()
        elem = tmp[i]
        tmp.remove(elem)
        tmp.insert(i, reg[0].tolist())
        tmp.insert(i+1, reg[1].tolist())
        tmp.insert(i+2, reg[2].tolist())
        tmp.insert(i+3, reg[3].tolist())
        R = np.array(tmp)
        
        # update indexes i_j
        tmp = i_j[i]
        ij = [tmp, [tmp[0], tmp[1]+size_QuadTree], [tmp[0]+size_QuadTree, tmp[1]],[tmp[0]+ size_QuadTree, tmp[1]+ size_QuadTree]]
        i_j.remove(i_j[i])
        i_j.insert(i,ij[0])
        i_j.insert(i+1,ij[1])
        i_j.insert(i+2,ij[2])
        i_j.insert(i+3,ij[3])
        
        # updates new indices in R
        indices_R.insert(i+1, indices_R[i])
        indices_R.insert(i+1+self.img.shape[1], indices_R[i])
        indices_R.insert(i+2+self.img.shape[1], indices_R[i])
        
        return R, i_j, dico_index_size, indices_R
        
    def trait_reg_thread(self, deb, fin, Di_for_Ri, s, o, angs, directs, k1, return_dict):
        '''
        Compute main computation, called by every thread, compute parameters of image compresion  :
             input : 
                            int : i = indice of region to split
                            int : deb = start of thread work in R
                            int : fin = end of thread work in R
                            array : Di_for_Ri = index of domain Di for region Ri
                            array : s = array contrast coefficient
                            array : o = array brightness coefficient
                            array: angs = array of angle coefficient
                            array : directs = array of direction coefficient
                            int : k = thread number
                            dict : return_dict = dict of returned values of threads
        '''
        
        #initialisation
        Di_for_Ri_t=list()
        directs_t=list()
        angs_t=list()
        s_t = list()
        o_t = list()
        
        # copy of shared values
        i_j = self.R_.i_j[deb:fin].copy()
        dico_index_size = self.R_.dico_index_size.copy()
        R = self.R_.R[deb:fin].copy()
        indices_R = self.R_.indices.copy()
        size = self.R_.dico_index_size[tuple(i_j[0])]
        
        
        # for all region 
        i=0
        while i < len(R):
            min_d = float('inf')
            R1 = R[i]
            if type(R1) == list:
                R1 = np.array(R1)
                
            #if new region size => dico_index_size is under previous then we compute new dictonary domain square
            if dico_index_size[tuple(i_j[i])] < size: 
                D_ = list_domain(self.img,dico_index_size[tuple(i_j[i])])
                D_.create_domains()
                D = D_.D
                if dico_index_size[tuple(i_j[i])] not in list(self.dico_D.keys()):
                    self.dico_D[dico_index_size[tuple(i_j[i])]] = D_
            
            # get the adequat dictionary of domain square
            D_ =  self.dico_D[dico_index_size[tuple(i_j[i])]]
            D=D_.D
            
            # for all domain square
            for k in range(len(D)):
                # Ri does not intersect with Di
                if indices_R[i] not in D_.indices[k]:
                    # compute parameters 
                    Dk = self.get_neighboors(D[k])
                    transformed_blocks= self.generate_transformed_D2(Dk)
                    # test all of paramters and keep the best <=> min distance Ri and Dk
                    for direct, angle, S in transformed_blocks:
                        si, oi = self.find_contrast_and_brightness2(R1, S)
                        S = si*S + oi
                        d = np.sum((R1 - S)**2)
                        if d < min_d:
                            min_d = d
                            Di_for_Ri_val = [k,D_]
                            s_val = si
                            o_val = oi
                            angles_val = angle
                            directs_val = direct
            # if distance between Ri and Dk is under tolerance or under size 8 keep parameters
            if min_d < self.tol or dico_index_size[tuple(i_j[i])] <= 2 :
                Di_for_Ri_t.append(Di_for_Ri_val)
                s_t.append(s_val)
                o_t.append(o_val)
                angs_t.append(angles_val)
                directs_t.append(directs_val)
            # else we split region
            else:
                R, i_j, dico_index_size,indices_R = self.split(i, R, i_j, dico_index_size, indices_R)
                i-=1
            i+=1
        self.R_.indices = indices_R
        return_dict[k1] = [Di_for_Ri_t, s_t, o_t, angs_t, directs_t, i_j, dico_index_size, R]
     
    def get_greyscale_image(self):
        '''Greyscale function'''
        return np.mean(self.img[:,:,:2], 2)
    
    def collection_subsquares(self):
        '''
        Main function, initiate thread and get back results
        '''
        #initialisation
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        R = self.R_.R
        D = self.D_.D
        #nb_cpu = multiprocessing.cpu_count()
        nb_cpu = 6
        nb_reg_per_thread = (len(R)//nb_cpu)
        Di_for_Ri= list()
        directs = list()
        angs = list()
        s = list()
        o = list()
        threads = []
        k=0
        
        # manages if number of task % numbers of thread != 0
        if len(R) % nb_cpu != 0:
            mod = len(R) % nb_cpu
            #create threads
            for i in range(0,len(R),nb_reg_per_thread):
                t = Process(target=self.trait_reg_thread, args=(i,i+nb_reg_per_thread, Di_for_Ri, s, o, angs, directs, k, return_dict))
                threads.append(t)
                k+=1
            nb = len(R) // nb_reg_per_thread
            t = Process(target=self.trait_reg_thread, args=(nb,nb+mod, Di_for_Ri, s, o, angs, directs, k, return_dict))
            threads.append(t)
            k+=1
        else :
            #create threads
            for i in range(0,len(R),nb_reg_per_thread):
                t = Process(target=self.trait_reg_thread, args=(i,i+nb_reg_per_thread, Di_for_Ri, s, o, angs, directs, k, return_dict))
                threads.append(t)
                k+=1
                
        #start threads
        for t in threads:
            t.start()
        
       # join threads and assemble computed values
        k1=0
        self.R_.i_j = []
        self.R_.dico_index_size = dict()
        self.R_.R = list()
        for t in threads:
            t.join()
            # append all results in lists
            Di_for_Ri.append(return_dict[k1][0])
            s.append(return_dict[k1][1])
            o.append(return_dict[k1][2])
            angs.append(return_dict[k1][3])
            directs.append(return_dict[k1][4])
            i_j = return_dict[k1][5]
            dico_index_size = return_dict[k1][6]
            self.R_.R.append(return_dict[k1][7])
            self.R_.i_j.append(i_j) 
            for key,val in dico_index_size.items():
                if key not in list(self.R_.dico_index_size.keys()):
                    self.R_.dico_index_size[key] = val
            k1+=1
        
        # assemble results
        tmp = Di_for_Ri
        Di_for_Ri = [item for sublist in tmp for item in sublist]
        tmp = s
        s = [item for sublist in tmp for item in sublist]
        tmp = o 
        o = [item for sublist in tmp for item in sublist]
        tmp = angs
        angs = [item for sublist in tmp for item in sublist]
        tmp = directs
        directs = [item for sublist in tmp for item in sublist]
        tmp = self.R_.i_j
        self.R_.i_j = [item for sublist in tmp for item in sublist]
        tmp = self.R_.R
        self.R_.R = [item for sublist in tmp for item in sublist]

        return Di_for_Ri, o, s, angs, directs
    
    def enhance_image(self):
        self.f1 = self.img-self.g # résidus
        self.f2 = np.where(self.f1 < 0, 0, self.f1)# ignore negtive values
        T = self.find_thresh(self.f2)
        self.f3 = np.where(self.f2 >= T, self.f2,0)
        
    def enhance(self):
        '''
        Main function :
           collection_subsquares = compute best parameters
           modeled_image = modeled the image
           And enhance_image = enhance image
        '''
        print(self.img.shape)
        
        # original image 
        fig = plt.figure()
        plt.imshow(self.img,cmap='gray')
        #plt.show() # decomment for notebook
        fig.savefig("results/image_original.jpg", bbox_inches='tight', pad_inches=0)
        
        # parameters
        print("Compress")
        Di_for_Ri, o, s, angles, directs = self.collection_subsquares()
        
        # modeled image
        print("Modeled")
        self.g=self.modeled_image(Di_for_Ri,o,s, angles, directs)
        fig = plt.figure()
        plt.imshow(self.g,cmap='gray')
        #plt.show() # decomment for notebook
        fig.savefig("results/image_modeled.jpg", bbox_inches='tight', pad_inches=0)
        
        # enhance image
        print("Enhance")
        self.enhance_image()
        fig = plt.figure()
        plt.imshow(self.f3,cmap='gray')
        #plt.show() # decomment for notebook
        fig.savefig("results/image_enhance.jpg", bbox_inches='tight', pad_inches=0)
        
        return self.f3
    
    def ti(self, fhat,oi,si):
        '''
        Compute ti
        '''
        return si*fhat+oi

    def modeled_image(self, Di_for_Ri, o, s, angles, directs, itere=10):
        '''
        Compute modeled image with parameters:
            input : 
                            array : Di_for_Ri = index of domain Di for region Ri
                            array : s = array contrast coefficient
                            array : o = array brightness coefficient
                            array: angs = array of angle coefficient
                            array : directs = array of direction coefficient
        '''
        # initialize randomly modeled image
        self.g = np.array([np.random.randint(0, 256, (self.img.shape[0], self.img.shape[1]))])
        self.g=self.g.reshape((self.size,self.size))
        
        # for itere iterations
        for i in range(itere):
            k2=0
            
            # for all region
            for reg in Di_for_Ri:
                
                #apply transformations to image
                indexD, D_ = reg 
                size = D_.size_square_R
                i_j = D_.i_j[indexD]
                i_j_R = self.R_.i_j[k2]
                ghat = self.apply_transformation(self.g[i_j[0]:i_j[0]+(2*size), i_j[1]:i_j[1]+(2*size)], directs[k2], angles[k2])
    
                #shape verification
                if ghat.shape[0] != ghat.shape[1]:
                    if ghat.shape[0] < ghat.shape[1]:
                        ghat = ghat[:,:ghat.shape[0]]
                    if ghat.shape[1] < ghat.shape[0]:
                        ghat = ghat[:ghat.shape[1],:]
                
                # compute modeled image
                im = self.get_neighboors(ghat)
                f=self.ti(im,o[k2],s[k2])
                
                # shape verification
                tmp = self.g[i_j_R[0]:i_j_R[0]+size,i_j_R[1]:i_j_R[1]+size]
                if tmp.shape[0] != tmp.shape[1]:
                    if tmp.shape[0] < tmp.shape[1]:
                        tmp = tmp[:,:tmp.shape[0]]
                    if tmp.shape[1] < tmp.shape[0]:
                        tmp = tmp[:tmp.shape[1],:]
                if f.shape != tmp.shape:
                    if f.shape[0] > tmp.shape[0]:
                        f = f[:tmp.shape[0],:tmp.shape[1]]
                    elif f.shape[0] < tmp.shape[0]:
                        tmp = tmp[:f.shape[0],:f.shape[0]]
                
                self.g[i_j_R[0]:i_j_R[0]+tmp.shape[0],i_j_R[1]:i_j_R[1]+tmp.shape[1]] = f
                k2+=1

        return self.g

    def find_thresh(self, f2):
        '''
        Find the optimal threshold : 
            input : 
                image : f2 enhance image with noise
        '''
        std_f2 = np.std(f2)
        T0 = 2.5*std_f2
        list_std = []
        for i in range(f2.shape[0]):
            for j in range(f2.shape[1]):
                if f2[i,j] < T0:
                    list_std.append(f2[i,j])

        std_noise= np.std(list_std)
        alpha=3
        return alpha*std_noise
    
    
    def rotate(self, img, angle):
        '''Rotate : 
                input :
                        image : img
                        float : angle
                output:
                        image
        '''
        return ndimage.rotate(img, angle, reshape=False)

    def flip(self, img, direction):
        '''Flip: 
                input :
                        image : img
                        float : direction
                output:
                        image
        '''
        return img[::direction,:]

    def apply_transformation(self, img, direction, angle):
        '''Apply transformation : 
                input :
                        image : img
                        float : direction
                        float : angle
                output:
                        image
        '''
        return self.rotate(self.flip(img, direction), angle)

    def generate_transformed_D2(self, img):
        '''Transform domains squares : 
            input : 
                    image : img
            output : 
                    list : transformed_blocks
        '''
        transformed_blocks = []
        directions = [1, -1]
        angles = [0, 90, 180, 270]
        candidates = [[direction, angle] for direction in directions for angle in angles]
        for direction, angle in candidates:
            transformed_blocks.append([direction, angle, self.apply_transformation(img, direction, angle)])
        return transformed_blocks

    def find_contrast_and_brightness2(self, R, S):
        '''
        Find optimal constrast and brightness for a region square R with transform square S:
            input : 
                    image : R
                    image : S
            output : 
                    float : s
                    float : o
        '''
        A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
        b = np.reshape(R, (R.size,))
        x, _, _, _ = np.linalg.lstsq(A, b)
        return x[1], x[0]
        
    def get_neighboors(self, img):
        '''
        get neighborhood of image 2x smaller:
            input : 
                image : img
            output : 
                image : img_neigh
        '''
        img_neigh = np.zeros((img.shape[0]//2,img.shape[1]//2))
        step = 2
        for i in range(img_neigh.shape[0]):
            for j in range(img_neigh.shape[1]):
                img_neigh[i,j] = np.mean(img[i*step:(i+1)*step,j*step:(j+1)*step]) 
        return img_neigh

t = time.time()

if not os.path.isdir(("{}/results".format(os.getcwd()))):
    os.mkdir("{}/results".format(os.getcwd()))
#img = np.array(Image.open('run/img_test.png'))
#img = np.array(Image.open('run/img_archive.png'))
img = np.array(Image.open('run/img_papier.png'))

#  you must choose a size size_square_R such that twice this value divides the whole image of size size so that the range square and the domains squares divides the image and multiple of 8.
# with X ranging from 1 to 10
# for example size_square_R = 8 and size = 32 : 8x2=16 and 32/16 = 2 ok
# size_square_R=8, tol=X, size=32 ect...
# size_square_R=8, tol=X, size=48 ect...
# size_square_R=8, size_square_R=16, tol=X, size=64 ect...
# size_square_R=8, size_square_R=16, tol=X, size=96 ect...
# size_square_R=8, size_square_R=16, size_square_R=32, tol=X, size=256 ect...
# size_square_R=8, size_square_R=16, size_square_R=32, tol=X, size=512 ect...
# ect...
enhance_im_ = enhance_img(img,size_square_R=8, tol=6, size=32)
enhance_im = enhance_im_.enhance()
print(time.time()-t)