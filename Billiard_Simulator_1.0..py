import math,itertools,textwrap
import numpy as np
import tkinter as tk
import tkinter.font as tkfont
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from tkinter import ttk
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

"""Math core"""
#Spaces
class HyperbolicSpace:
    
    """Difines the geometric concepts associated with hyperbolic like distances, angles, inner products and transformations between coordinates or tangent planes."""
    
    def __init__(self,d=2):
        self.dim=d #space dimension
        self.base=self.base_vectors() #Associated base (normally in Hiperbolic or Poincare Ball)
    
    #Vector spaces propierties and hiperbolic space properties (euclidean things are useful because of the embeding)(hyperbolic space in Poincare coords)
    def base_vectors(self,dim=None):
        
        """Extracts canonical ortonormal base of a certain dimension space"""
        
        if dim==None:
            dim=self.dim
        basis = np.zeros((dim, dim))
        for i in range(dim):
            basis[i, i] = 1.0
        return basis
    def planeq_to_comp(self,func,*args,dim=None):
    
        """If you have a plane defined by an linear equation with function "func" of the form f(x1,...,xn,*args)=0,extracts normal vector (a) and off center deviation (b)"""
        
        if dim is None: #In absence of dimension acts over the dimension of the defining space
            dim=self.dim
            
        #Extracts the components of the linear equation of a plane where a=[a0,a1,...,an] i.e. f(x)=a^T*x-b=0
        b=func(np.zeros(dim),*args)
        base=self.base_vectors(dim)
        a=[]
        for base_vect in base:
                a.append(func(base_vect,*args)-b)
        
        return np.array(a),np.array(b)
    def euclidean_distance(self,p1,p2):
        """Returns euclidean distance between two points p1 and p2."""
        return np.linalg.norm(p1 - p2)
    def euclidean_norm(self,p1):
        """Returns euclidean norm of the point p1."""
        return np.linalg.norm(p1)
    def euclidean_in_prod(self,p1,p2):
        """Returns the euclidean inner product of two vectors p1,p2."""
        return np.dot(p1, p2) 
    def mink_in_prod(self,p1,p2):
        """Returns Minkowsky inner product between two vectors p1 and p2."""
        return -p1[0]*p2[0] + np.dot(p1[1:], p2[1:])
    def mink_norm2(self,p1):
        """Returns Minkowsky norm squared of the vector p1."""
        return self.mink_in_prod(p1, p1)
    def de_witt_vect_in_prod(self,p1,p2):
        
        """Returns the DeWitt inner product between two vectors p1 and p2."""
        dim=len(p1)
        G_METRIC=np.eye(dim, dtype=float) - np.ones((dim, dim), dtype=float)
        
        return p1@G_METRIC@p2
    def de_witt_form_to_vect(self,p1):
        """Transforms a form into a vect on dual space of DeWitt"""
        dim=len(p1)
        G_INV=np.linalg.inv(np.eye(dim, dtype=float) - np.ones((dim, dim), dtype=float))
        
        return G_INV@p1
    def hyperbolic_distance(self,p1,p2):
        
        """Returns the distance between two points in te Poincare Ball space"""
        
        isometric_invariant=(2*(self.euclidean_distance(p1,p2))**2)/((1-self.euclidean_norm(p1)**2)*(1-self.euclidean_norm(p2)**2)) #It does not change under isometries
        try:    
            return math.acosh(1+isometric_invariant)
        except ValueError:
            print(f"The points {p1} or {p2} has to be in the Poincare plane")
    def hyperbolic_norm(self,p1,d=None):    
        
        """Returns the norm of p1 in Poincare Ball space."""
        
        if d==None:
            d=np.zeros(self.dim)
        
        return (2*self.euclidean_norm(p1))/(1-self.euclidean_norm(d)**2)
    def hyperbolic_in_prod(self,p1,p2,d=None):
        
        """Returns the Poincare Ball inner product of two vectors p1,p2 that are not recesarly at distance d zero"""
        
        if not isinstance(d,np.ndarray):
            d=np.zeros(self.dim)
        
        return (4*self.euclidean_in_prod(p1,p2))/((1-self.euclidean_norm(d)**2)**2)
    def angles(self,p1,p2):
        
        """Returns the angles between vectors p1 and p2 in the Poincare Ball representation"""
        
        u,v=p1/self.hyperbolic_norm(p1),p2/self.hyperbolic_norm(p2)
        return math.acos(self.hyperbolic_distance(u,v))
    def ortogonality_and_normality(self,vectors):
        
        """It calculates the Gram matrix and veryfies if the matrix is diagonal and if its unitary and thus if the vectors are ortogonals and normals"""
        
        G=[]
        for i in range(len(vectors)):
            for j in range(len(vectors)):
                G.append(self.hyperbolic_in_prod(vectors[i],vectors[j]))

        return np.all(np.abs(G - np.diag(G)) < 1e-12), np.all(np.abs(np.diag(G)) == 1.0) #Comprueba si todos los elementos de G sin traza son zero
    
    #Transformations(in Poincare coords)
    def moebius_add(self,p1,p2):
        
        """Möbius add isometry in vector representation is a difeomorphism (continuos and biyective transform between manifolds f:M->N) that mantains the metric invariant(isometry)(f:M->M) and in this case can transport vectors to the point p1 and sum it. For inverse transform just has to aply -p1. It forms a gyrogroup"""
        
        numerator=(1-self.euclidean_norm(p1)**2)*p2+(1+self.euclidean_norm(p2)**2)*p1+2*self.euclidean_in_prod(p1,p2)*p1
        denominator=1+self.euclidean_norm(p2)**2 * self.euclidean_norm(p1)**2+ 2 * self.euclidean_in_prod(p1,p2)
        
        return np.array(numerator / denominator)
    def exp0(self,v1):
        """The Riemann exponential at 0 is a transformation that maps vector v from the tangent space of origin to the hyperbolic space (exp_0:T_0M->M)."""
        return np.tanh(self.hyperbolic_norm(v1)/2)*(v1/self.euclidean_norm(v1))
    def expx(self,p1,v1):
        """The Riemann exponential at p1 is a transformation that maps the vector v from the tangent space of point p1 to the hyperbolic space (exp_p1:T_p1M->M)."""
        return self.moebius_add(p1,self.exp0(v1))
    def log0(self,v1):
        """Inverse of the riemann exponential map at 0. Returns the vector in the tangent space from a vector in the hiperbolic space log0:M->T_0M"""
        return np.arctanh(self.euclidean_norm(v1)/2)*(v1/self.euclidean_norm(v1))
    def logx(self,p1,v1):
        """Inverse of the riemann exponential map at p1. Returns the vector in the tangent space of p1 from a vector v1 in the hiperbolic space log_p1:M->T_p1M"""
        return (1-self.euclidean_norm(p1)**2)*self.log0(moebius_add(-p1,v1))
    def parallel_transp(self,p1,p2,v1):
        """Möbius pushforward is the first derivative of the moebius isometry and considered the lineal Jacobian and a paral·lel transport transformation. Allows to transport vector fields v from point p1 to p2 perserving geometry. Thus has the form dM_p1(p2):T_{p2}M->T_{M_p1(p2)M} that can be writed as P_(x->y):T_xM->T_yM."""
        return logx(p1,moebius_add(p1,expx(p2,v1)))
    
    #Coords transformations
    def scale_to_mink_matrix(self,dim=None):
        
        """Returns the transformation matrix of scale factors coordinates (DeWitt space) to Minkowksy space"""
        
        if dim is None:
            dim=self.dim
        G =(np.eye(dim)-np.ones((dim, dim))) #Generates DeWitt supermetric (with normalizing factor)
        eigvals, eigvecs=np.linalg.eigh(G) #Diagonalizices supermetric to get transformation vectors where G is diagonal (and therefore Minkowsky like)
        if np.sum(eigvecs[:, 0]) < 0: #Rearanges proper vectors to have the negative one first
            eigvecs[:, 0] *= -1
        
        scale_matrix=np.diag(np.sqrt(np.abs(eigvals)))
        return scale_matrix@eigvecs.T
    def scale_to_poinc(self,betas):
    
        """Transforms scale factors to poincare coordinates. This is done by diagonilizing the supermetric (DeWitt space) to Minkowsky space, projecting Minskowski space to Unit Hiperboloid an then transforming to Poincare Ball coords by doing an Stereogrphic Projection."""
        
        X=[]
        
        betas_barr=self.scale_to_mink_matrix(dim=len(betas))@betas 
        
        if self.mink_norm2(betas_barr)<-1e-9: #Beta is on the time cone
            
            #It does not transform to gamma space because is less stable computationally. Instead it does the stereographic projection with minkoski space
            
            rho = np.sqrt(np.abs(self.mink_norm2(betas_barr))) #Sets hiperbolic radi of the unit hiperboloid
            
            if abs(betas_barr[0]+rho) < 1e-9: #Numerical control error
                return np.zeros_like(len(betas_barr))
            else: #Fallback
                return betas_barr[1:]/(betas_barr[0]+rho)
        elif self.mink_norm2(betas_barr)<=1e-9: #Beta is proper to infinity (light cone)

            if abs(betas_barr[0]) < 1e-12: #Numerical control error
                return np.zeros(len(betas_barr))
            else: #Fallback
                return betas_barr[1:]/betas_barr[0]
        else: #Beta is out the time cone 
            return np.zeros_like(len(betas_barr))
    def vect_scale_to_poinc(self, betas, betas_vel):
        dim = len(betas)
        T = self.scale_to_mink_matrix(dim)
        
        #First pushforward of diagonal transformation wich is another diagonal transformation
        Y = T @ betas
        V_Y = T @ betas_vel
        
        #Second pushforward of minkowski coordinates to hiperboloid coordinates
        y_norm_sq = self.mink_norm2(Y)
        if y_norm_sq >= -1e-9:
            return np.zeros(dim - 1)  
        rho = np.sqrt(np.abs(y_norm_sq))
        y_dot_v = self.mink_in_prod(Y, V_Y)
        rho_dot = -y_dot_v / rho
        
        #Third pushwforward from hiperboloid to Poincare Ball
        Y0 = Y[0]
        Y_vec = Y[1:]
        V0 = V_Y[0]
        V_vec = V_Y[1:]
        denom = Y0 + rho
        if abs(denom) < 1e-9:
            return np.zeros(dim - 1)
        numerator = V_vec * denom - Y_vec * (V0 + rho_dot)
        vel_poinc = numerator / (denom**2)
        
        return vel_poinc
    def scale_plane_to_poinc_vects(self,func,*args,dim=None):
        
        """Returns Base and Offset needed to define a Hyperplane in the Poincare coordinates by having the function "func" in Minkowsky space that defines the plane f(x1,...,xn,*args)=0"""
        
        a,b=self.planeq_to_comp(func,*args,dim=dim) #Extracts normal vector "a" and distance to zero "b"
        a=a@np.linalg.inv(self.scale_to_mink_matrix(dim=dim))#Transforms to minkowski knowing "a" is 1-form
        
        #Is usefull to know that a plane in Scale or Minkowski coordinates of form a*x+b=0 in the stereographic projection is a_0*|x|^2+2*a_space*x+b+a_0. If a_0=0 is a line or plane in Poincare Ball, else is a sphere.
        
        if self.euclidean_norm(a[1:])<1e-12: #Fallback if normal vector is not timelike
            print(f"Not timelike cone finded in:{a}")
            return None, None
        if abs(a[0])<1e-12: #The plane is a centered hyperplane in the euclidean embeding of the Poincare Ball
            offset=np.zeros_like(a[1:])
        else:  #The plane is a hypersphere in the euclidean embeding of the Poincare Ball
            offset=a[0]/(self.euclidean_norm(a[1:])+np.sqrt((self.euclidean_norm(a[1:]))**2-(a[0])**2))*(a[1:]/self.euclidean_norm(a[1:]))
        base_matrix=null_space(a[1:].reshape(1, -1)).T
        
        return base_matrix, offset


    #Geodesics
    class Geodesic():

        """Geodesic computational object that can be used to calculate points of a geodesic or define and project to plane. It is defined with two points. """
        
 
        def project_to_plane(self,p1,p2):
        
            """Projects 2 vectors (p1,p2) on the euclidean embeding into de base vectors(self.u,self.v) and returns the projected vectors(p1_prima,p2_prima)"""
            
            p1_prima=np.array([self.space.euclidean_in_prod(p1,self.u),
                                   self.space.euclidean_in_prod(p1,self.v)])
            p2_prima=np.array([self.space.euclidean_in_prod(p2,self.u),
                                   self.space.euclidean_in_prod(p2,self.v)])  
            return p1_prima,p2_prima
        def is_diameter_or_cicular(self):
            
            """It considers if the system of points forms an indetermiate equation system and, in consequence, cannot form a geodesic plane wich means that the geodesic is a straight line"""

            return True if all(x is None for x in self.v)==None else False
        def _radius_and_center(self):      

            """Calculates the radius R and the center C of the circle generated by p1 and p2 that is ortogonal to the boundry of the hyperbolic space |x|<1 """
        
            if self.is_diameter_or_cicular():
                return None, None #The radius is infinite and thus have a straight line
            else:
                #Everything is projected to work only in 2D
                p1_prima, p2_prima=self.project_to_plane(self.p1,self.p2)
                
                #It calculates c resolving |p1-c|^2=R^2, |p2-c|^2=R^2 (plus the ortogonality condition described in the next comment)
              
                D=np.array([[p1_prima[0],p1_prima[1]],
                            [p2_prima[0]-p1_prima[0],p2_prima[1]-p1_prima[1]]])
                r=np.array([(self.space.euclidean_norm(p1_prima)+1)/2,
                (self.space.euclidean_norm(p2_prima)-self.space.euclidean_norm(p1_prima))/2])
                
                c2D=np.linalg.inv(D)@r
                
                c=c2D[0]*self.u+c2D[1]*self.v
                
                #The radius is thus calculated considering both intersection points of an ortogonal intersection of the circle and the boundry of space solving x_int*(x_int-c)=0,|x_int|^2=1 y |x_int-c|=R wich gives the ortogonal condition |C|-R^2=1
                
                R=self.space.euclidean_norm(c)-1.0
            
            return R, c
        def _points(self,n_points=300):
        
            """Returns a colection of points inside the geodesic """
            
            if self.is_diameter_or_cicular(): #Makes one method or other based on if it is a line or a cirlce
                t=np.linspace(-1,1,n_points)
                return t[:,None]*self.u
            else:
                #Makes initial and final angle conditions to fit into the Poincare Ball
                p1_prima, p2_prima=self.project_to_plane(self.p1,self.p2)
                phi_i=np.arctan2(p1_prima[1], p1_prima[0])
                phi_f=np.arctan2(p2_prima[1], p2_prima[0])
                 
                #Makes an angular parametrization to generate a colection of points
                t=np.linspace(0,1,n_points)
                phi=phi_i+t*(phi_f-phi_i)
                
                return self.center+self.radius*(np.cos(phi)[:,None]*self.u+np.sin(phi)[:,None]*self.v)
        def __init__(self, space,p1, p2, n_points=300):
            self.space=space
            self.p1=p1
            self.p2=p2
            self.u,self.v= self._plane_vectors()
            self.radius,self.center=self._radius_and_center()
            self.points=self._points(n_points)    

    #Hypersuperfaces
    class GeodesicHyperplane():

        """Computational object that represents a Geodesic Hyperplane defined with its vectors and the offset to center in the Poincare Ball representation. It generates a collection of points of a geodesic submanifold of the hyperbolic space. Can also be used to project vectors or find its normal to plane components and can extract representation-like things like its border and grid."""
        
        def vectors_are_base(self,vectors):
            """It looks if all the vectors are linearly independent. It cannot look if they generate all of hyperplane geometry because it is generated by the vectors"""
            return np.linalg.matrix_rank(vectors) == vectors.shape[0] 
        def base_vectors(self,vectors):
            """Realizes Gram-Shcmidt generalized to the dimension to extract base vectros from the array vectors"""
            u_ortogonal=[vectors[0]/self.space.euclidean_norm(vectors[0])]
                
            for v in vectors[1:]:
                u_temp=v.copy()
                for u in u_ortogonal:
                    proj_vi=(self.space.euclidean_in_prod(u_temp,u)/self.space.euclidean_in_prod(u,u))*u
                    u_temp -= proj_vi
                if self.space.euclidean_norm(u_temp)>1e-12:#Evades degenerate basis
                    u_temp=u_temp/self.space.euclidean_norm(u_temp)
                    u_ortogonal.append(u_temp)

            return np.array(u_ortogonal)       
        def project_to_hyperplane(self,vectors):
            
            """Project a colection of vectors into the plane basis and it returns the result tangent_vectors"""
            
            tangent_vect=[]

            for vect in vectors:
                tangent_vect.append(np.array([self.space.euclidean_in_prod(vect,self.base[i])/self.space.euclidean_in_prod(self.base[i],self.base[i]) for i in range(self.sub_dim)]))
            
            return np.array(tangent_vect)   
        def normal_projection(self,vectors):
            """Project a colection of vectors outo the plane basis and it returns the result of the normal projection"""
            return vectors-self.project_to_hyperplane(vectors)  
        def _is_sphere_or_plane(self):
            """Returns True if the hyperplane is a plane in the euclidean embeding and false elsewhere"""
            return np.all(np.abs(self.offset) < 1e-12)
        def _normal_vector(self):
            """Gives the normal vector of the plane"""
            U,S,V_h=np.linalg.svd(self.base)
            return V_h[-1]
        def _is_in_plane(self,vector):
            """Gives True if the given vector is in hyperplane and gives false otherwise"""
            return True if self.normal_projection(np.array([vector]))<1e-12 else False
        def hyperplane_points(self,n_points=100):
   
            """Returns a colection of points in the hyperplane of a certain base vectors and offset with intrinsic geometry functions"""
   
            # The vector v is the set of vectors belonging to the hyperplane (v\inU), each of which lives in the tangent space of the plane's origin p0 (v\in T_p0M), and the alphas are their linear combinations
            alpha=np.array([np.linspace(0.5,-0.5) for i in range(self.sub_dim)])

            v = []
            for a in alpha.T:
                v.append(np.dot(a,self.base))
            v = np.array(v)
            for vect in v: #maps each point into the position of the offset.
                vect=self.space.moebius_add(-self.offset,vect)  
            return vect
        def _radius_and_center(self):  
            
            """Calculates the radius R and the center C of the hipersphere generated by the base vectors u_i and the offset that is ortogonal to the boundry of the hyperbolic space |x|<1. Uses the embeding and returns None if its centered."""
            
            if self._is_sphere_or_plane():
                return None,None
            else:
                #One must consider that the hypersphere has to be ortogonal to the Poincare Ball so that means that there is the ortogonal condition |c|^2=R+1
                #One also has to consider the definition of the points in the sphere are defined by |offset-c|^2=R^2
                #Finally, one knows that the base vectors in T_pM are tangent to the sphere and knowing that the vectors (offset-c) are normal to the sphere one has the ortogonal condition (offset-c)*u_i=0.
                #With the last two conditions it is possible to obtain the center of the sphere by solving a set of equations and with the first one it is possible to obtain the radius.
                
                U= np.vstack(self.base) #Base matrix
                b_tan = np.array([self.space.euclidean_in_prod(self.offset,u) for u in self.base]) #offsets projected
                
                #Adds circle belonging condition
                U_aug=np.vstack([U,self.offset])
                b_aug=np.append(b_tan,(self.space.euclidean_in_prod(self.offset,self.offset)+1)/2)
                
                c=np.linalg.inv(U_aug) @ b_aug
                R=np.sqrt(self.space.euclidean_norm(c)-1.0)
                
                return c, R   
        def get_intesect_points(self, plane,n_points=100):
            
            """Gives the points that intersects this geodesic hyperplane if has dimension 2 and another geodesic hyperplane of arbitrary dimension. It is used to calculate the geodesic points of the slices with the walls and uses intrinsic geometry functions to calculate the points."""
            
            if self.sub_dim==2:
                if self._is_sphere_or_plane():
                    #Lineal Plane (Simetry Walls)
                    if plane._is_sphere_or_plane():
                        #If we define the given hyperplane with its normal vector n^T*x=0 and then if this plane has x=A*u where u is the component matrix and A the basis matrix one can find the intersection components by the equation n^T*A*u=0.
                        
                        #First it calculates the normal vector of the given hyperplane with a svd.
                        U,S,Vt=np.linalg.svd(plane.base)
                        n=Vt[-1]
                        
                        #Defining r=n^T*A then 
                        r1=float(n@self.base[0])
                        r2=float(n@self.base[1])
                        if abs(r1)<1e-12 and abs(r2)<1e-12:
                            print("Both planes are coplanar")
                            return None
                        else:
                            #Solves the system and normalizes de vector u to know the direction x0
                            u=np.array([-r2,r1])
                            x0=self.space.euclidean_in_prod(u,self.base)
                            x0/=self.space.euclidean_norm(x0)
                            
                            #Limits the intersection to the boundry of the poincare ball
                            t=np.linspace(-1,1,n_points)
                            return t[:,None]*x0
                    #Sphere plane (Gravitational and Electromagnetic wall)
                    else:
                        offset_nd = plane.offset
                        u, v = self.base[0], self.base[1]
                        
                        #Scalar product to know the direction of the offset component in the plane
                        coeff_u = self.space.euclidean_in_prod(offset_nd, u)
                        coeff_v = self.space.euclidean_in_prod(offset_nd, v)
                        
                        #Projected vector on the slice in the global space
                        pc = coeff_u * u + coeff_v * v
                        norm_pc = np.linalg.norm(pc)

                        dir_perp = -coeff_v * u + coeff_u * v
                        dir_perp /= np.linalg.norm(dir_perp) 
                        
                        #Gets a lienar linspace to get points first 
                        t = np.linspace(-0.999, 0.999, n_points)
                        #Makes a colection of points in the normal direction     
                        points_at_origin = t[:, None] * dir_perp 
                            
                        X = []
                        for p in points_at_origin:
                            X.append(self.space.moebius_add(-pc, p)) #transposes all points to global space
                            
                        return np.array(X)
                else:
                    print("The method intesection_points is valid only if the plane has offset 0.")
            else:
                print("The method intesection_points is only valid if this plane dimension is 2")
        def get_border(self,n_points=100):
            
            """Given a certain dimension, returns a colection (with n_points) of points in the subspace border. This method just works on 2D but can be generalized"""
            
            if self.sub_dim==2:
                #The border, as well as the grid, is calculated by intersecting a sphere(in this case the unit sphere) with the geodesic plane in center an the move it to the offset point
                
                points=[]
                #If you parametrize the u,v components in angular coordinates it can be expresed just in term of phi.
                phi=np.linspace(0,2*np.pi,int(n_points))
                components=np.array([np.cos(phi),np.sin(phi)])

                for comp in components.T:
                    points.append(self.space.moebius_add(-self.offset,self.space.euclidean_in_prod(comp,self.base)))

                return np.array(points).T
            else:
                print("The border just can be shown if the dimension is 2")     
        def get_grid(self,dr=0.25,dphi=2*np.pi/8,n_points=100):
        
            """Given a certain dimension, returns a colection (with n_points) of points of a grid separed by dr and dphi.It is convinient for better drawing to have 1/dr and 2pi/dphi whole number. This method just works on 2D but can be generalized"""
            
            if self.sub_dim==2:
                
                #The grid is maked by intersecting diferent hiperplanes with a plane in the center an the move it to the offset.The intersectin planes are a colection of hyperspheres centered in origin with diferent radius separed each one by dr generating the isoradial circles.And also has to intersect with geodesic hiperplanes that has one tangent vector on this plane and the normal vector can be parametrized to depend on the angle giving the isoangular lines.
                
                grid_lines=[]
                #The intesection of spheres and this plane gives a circle equation with center components c_i=-<p0,u_i>, and radius R^2=rho^2-p0^2+c^2 and (u-c_x)^2+(v-c_y)^2=R^2 where u and v are the posible values that want and rho=tanh(r/2).
                
                r=dr
                rho=np.tanh(r/2)
                while rho<1: #Goes over the colection of all spheres
                    rho=np.tanh(r/2)#Radius calculus

                    #If you parametrize the u,v components in angular coordinates it can be expresed just in term of phi.
                    phi=np.linspace(0,2*np.pi,int(n_points/2))
                    components=np.array([rho*np.cos(phi),rho*np.sin(phi)])
                    
                    #Creates a colection of points
                    circle=[]
                    for comp in components.T:
                        circle.append(self.space.moebius_add(-self.offset,self.space.euclidean_in_prod(comp,self.base)))
                    grid_lines.append(circle)
                    r+=dr
                
                #One can show that the intersection of two geodesic planes is always a geodesic and in this case the geodesic passes along the center of the current plane because the cutting plane always passes through the center. This means that the cuting lines are simply a rect line.
                
                phi=0
                while phi-2*np.pi<10e-12:
                    
                    R=np.linspace(0.0,1.0,int(n_points/2))
                    rect=[]
                    for r in R:
                        rect.append(self.space.moebius_add(-self.offset,r*np.cos(phi)*self.base[0]+r*np.sin(phi)*self.base[1]))
                    grid_lines.append(rect)
                    phi+=dphi
                return np.array(grid_lines)
                
            else:
                print("The grid just can be shown if the dimension is 2")     
        def get_eucl_dist(self,p1):
            
            """Gets distance from the point p1 to the plane in the euclidean embeding"""
            
            if self._is_sphere_or_plane():
                return np.dot(self.normal_vect, p1)
            else:
                if self.center is None or self.radius is None:
                    return 0.0
                dist_to_center =self.space.euclidean_norm(p1 - self.center)
                return dist_to_center-self.radius
        def __init__(self,space,vectors,offset=None):
            
            vectors=np.array([vectors[i] for i in range(len(vectors))]) #in Tp0M
            self.space=space
            if not offset is None:
                self.offset=np.array(offset)
            else:
                self.offset=np.zeros(self.space.dim)
            if self.vectors_are_base(vectors) and self.space.ortogonality_and_normality(vectors)[0] and self.space.ortogonality_and_normality(vectors)[1]:
                self.base=np.array(vectors)
                self.sub_dim=np.array(vectors.shape[0])
            else:
                self.base=self.base_vectors(vectors)
                self.sub_dim=len(self.base)
            self.normal_vect=self._normal_vector()
            self.center,self.radius=self._radius_and_center()

#Pysics Core (objects in the scale factor representation that has phisical meaning)
class Wall(HyperbolicSpace.GeodesicHyperplane):
    
    """Geodesic hyperplane in the scale factor representation"""
    
    def __init__(self,space,func,*args):
        self.vectors,self.offset=space.scale_plane_to_poinc_vects(func,*args,dim=space.dim+1)
        if not self.vectors is None and not self.offset is None:
            super().__init__(space,self.vectors,self.offset)
        self.normal,self.b=self.space.planeq_to_comp(func,*args,dim=space.dim+1)   
class Particle():
    def __init__(self,space,init_pos,init_kasner_exp,tau=0.0):
        self.space=space
        self.init_pos=np.array(init_pos,dtype=np.longdouble)
        self.init_vel=np.array(init_kasner_exp,dtype=np.longdouble)
        self.pos=np.array(init_pos,dtype=np.longdouble)
        self.kasner_vel=np.array(init_kasner_exp,dtype=np.longdouble)
        self.tau=tau
    def update(self, tau_target, walls):
        """Manages movement and colision of the particle to a cenrtain time tau_target with all walls"""
        remaining_tau = tau_target
        tolerance = 1e-12 #Numerical protection tolerance
        max_bounces = 100 #Max bounces to not explode in corners
        
        for _ in range(max_bounces):
            if remaining_tau <= 0:
                break

            nearest_collision_time = float('inf')
            collision_wall = None

            #Detects colisions
            for wall in walls:
                #Sets distance and velocity to wall 
                dist = np.dot(wall.normal, self.pos) - wall.b
                vel_proj = np.dot(wall.normal, self.kasner_vel)

                if vel_proj < -1e-12: #If projected velocity is small is near to wall 
                    t_col = -dist / vel_proj #Sets time of colision
                    if -tolerance < t_col < nearest_collision_time: #If time of colision is less than las time of colision means that this wall is closer than the other walls
                        nearest_collision_time = t_col
                        collision_wall = wall
            #Moving and reflecting of the particle
            if collision_wall and nearest_collision_time <= remaining_tau: #There is colision
                #Sets particle position near the wall
                step = max(0.0, nearest_collision_time)
                self.pos += self.kasner_vel * step
                self.tau += step
                remaining_tau -= step
                
                #Reflects velocity
                self._reflect_vel(collision_wall.normal)
                
                #Makes a Nudge to prevent transpassing the wall by errors of tolerance
                nudge_dir = self.space.de_witt_form_to_vect(collision_wall.normal)
                norm_nudge = np.linalg.norm(nudge_dir)
                if norm_nudge > 0:
                    self.pos += (nudge_dir / norm_nudge) * tolerance * 10
            
            else:#If there hasn't been colision changes particle position as normal
                self.pos += self.kasner_vel * remaining_tau
                self.tau += remaining_tau
                remaining_tau = 0
                break
    def _reflect_vel(self, w_form):
        """Makes the reflection of velocity v' = v - 2 * (w . v) / (w . w) * w^subido"""
        #Makes normal form into a vector
        w_vec = self.space.de_witt_form_to_vect(w_form)
        
        #Scalar products
        w_dot_v = np.dot(w_form, self.kasner_vel)
        w_dot_w = np.dot(w_form, w_vec)
        
        #Aply reflexion
        if abs(w_dot_w) > 1e-15: #Domain error protection
            self.kasner_vel = self.kasner_vel - 2 * (w_dot_v / w_dot_w) * w_vec
    def get_position_at_time(self, target_time, walls):
        
        """Estimates the exact position in arbitrary time."""

        self.tau = 0.0
        self.pos = self.init_pos.copy() 
        self.kasner_vel = self.init_vel.copy()
        
        time=np.linspace(0,target_time,100)
        for t in time:
            self.update(t, walls) #Updates position at this time
        
        return self.pos
#Simulation Core
class SimulationCore():
    
    """Simulation Object. Contains all planes and particles that can be added aswell as the space. Has update function to update its informatio in time and can get certain information from it to make graphics like border and grid of slices,particle position, etc."""
    
    def __init__(self,init_parameters):

        self.beta_dim=len(init_parameters["Initial Kasner Exp"])
        self.dim=self.beta_dim-1
        
        self.space=HyperbolicSpace(self.dim)
        
        self.tau=init_parameters["Initial Time"]
        #Creates all the slices by adding subspaces of that slices
        self.slices=[]
        for i,base_vect in enumerate(self.space.base):
            if i!=0: #Each silce is made by the base vector [1,0,0,...] and its other combinations
                self.slices.append(self.space.GeodesicHyperplane(self.space, [self.space.base[0],base_vect],np.zeros(self.dim)))
        
        #Creates all symetry walls
        self.symetry_walls=[]
        for a in range(self.beta_dim):
            for b in range(self.beta_dim):
                if a<b:
                    self.symetry_walls.append(Wall(self.space,self.symetry_plane,a,b))
        #Creates gravitational walls
        self.grav_walls=[]
        for a in range(self.beta_dim):
            for b in range(self.beta_dim):
                for c in range(self.beta_dim):
                    if a!=b and b!=c and c!=a:
                        self.grav_walls.append(Wall(self.space,self.grav_plane,a,b,c))

        #Creates p-form walls
        self.p_form_walls = []
        if init_parameters.get("P-Forms", False):

            p_list = init_parameters.get("P-Form List", [])
            couplings = init_parameters.get("Coupling Constants", [])
            
            spatial_dim = self.beta_dim - 1 if init_parameters.get("Dilaton", False) else self.beta_dim
            
            for p, coupling in zip(p_list, couplings):
                #Electric walls 
                for indices in itertools.combinations(range(spatial_dim), p):
                    self.p_form_walls.append(Wall(self.space, self.p_form_elec_plane, indices, coupling))
                #Magnetic walls
                p_magn = spatial_dim - p - 2
                if p_magn > 0:
                    for indices in itertools.combinations(range(spatial_dim), p_magn):
                        self.p_form_walls.append(Wall(self.space, self.p_form_magn_plane, indices, coupling))                
         
        #Creates Wall array
        raw_walls = np.concatenate([self.symetry_walls, self.grav_walls,self.p_form_walls])
        
        #Dropping equal walls 
        unique_walls = []
        seen_normals = []
        for w in raw_walls: #Looks if the normal is the same
            is_duplicate = False
            for seen in seen_normals:
                if np.allclose(w.normal, seen, atol=1e-9):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_walls.append(w)
                seen_normals.append(w.normal)
        self.walls = np.array(unique_walls)
        
        #Creates the particle and starts it in the initial setted time
        self.particle=Particle(self.space,init_parameters["Initial Beta Pos"],init_parameters["Initial Kasner Exp"],self.tau)
        self.particle.get_position_at_time(self.tau,self.walls)    
    def p_form_elec_plane(self, betas, indices, coupling=0.0):
        
        """Given the position betas, a list of indices defining the p-form, and the coupling constant, returns the point in the corresponding electric wall."""
        
        val = sum(betas[i] for i in indices)
        if abs(coupling) > 1e-12:
            phi = betas[-1]
            val -= 0.5 * coupling * phi
        return val   
    def p_form_magn_plane(self, betas, indices, coupling=0.0):
        """Given the position betas, a list of dual indices, and the magnetic coupling constant, returns the point in the corresponding magnetic wall."""
        
        val = sum(betas[i] for i in indices)
        if abs(coupling) > 1e-12:
            phi = betas[-1]
            val += 0.5 * coupling * phi
        return val
    def symetry_plane(self,betas,a,b):
        
        """Given certain position and a,b returns the a point in the corresponding simetry plane"""
        
        if a<b:
            return betas[b]-betas[a]
        else:
            raise ValueError("Value a must be less than b.")
    def grav_plane(self,betas,a,b,c):
    
        """Given certain position and a,b, returns the a point in the corresponding gravitational plane"""
    
        if a!=b and b!=c and c!=a:
            alfa=2*betas[a]
            for e in range(self.beta_dim):
                if e!=a and e!=b and e!=c:
                    alfa+=betas[e]
            return alfa
        else:
            raise ValueError("Values a,b,c must differ between them")
    def update(self,target_time,n_slice):
        
        """Updating of the time of the simulation as well as particle position"""
        
        self.tau=target_time
        self.particle.update(self.tau,self.walls)
        return self.get_part_pos(n_slice)
    def get_border(self,n_slice):
        """Gets a collection of points in the border of the correspondent slice n_slice in Poincare Ball coordinates"""
        return self.slices[n_slice].project_to_hyperplane(self.slices[n_slice].get_border().T)
    def get_grid(self, n_slice):
        
        """Gets a collection of point in the grid of the correspondent slice n_slice in Poincare Ball coordinates"""
        
        grid_lines=self.slices[n_slice].get_grid()
        projected_grid_lines=[]
        for line in grid_lines: #Projects all grid lines into the correspondent slice
            projected_grid_lines.append(self.slices[n_slice].project_to_hyperplane(line)) 
        return np.array(projected_grid_lines)
    def get_slice_def(self,n_slice):
        """Gets base and offset vectors that defintes the slice n_slice in Poincare Ball coordinates"""
        return self.slices[n_slice].base,self.slices[n_slice].offset
    def get_part_pos(self,n_slice):
        """Gets the particle position projected in the slice n_slice in Poincare Ball coordinates"""
        return self.slices[n_slice].project_to_hyperplane([self.space.scale_to_poinc(self.particle.pos)])[0]
    def get_walls(self,n_slice):
        """Gets the intersection points of a wall object with the slice n_slice in Poincare Ball coordinates."""
        projected_walls=[]
        for wall in self.walls: #Gets all intersected points of wall and slice
            points = self.slices[n_slice].get_intesect_points(wall)
            if points is  not None:
                projected_walls.append(self.slices[n_slice].project_to_hyperplane(points)) #Projects point into slice
        return projected_walls
#Utilities
class Graphic():

    """Core graphic that resumes all information on creation of matplotlib graphics"""

    def __init__(self,root,
                 data,
                 init_time=0.0,
                 wall_data=None,
                 update=None,
                 lims=[None,None],
                 title="",
                 axis_titles=["",""],
                 grid=False,
                 plot_type="plot",
                 color="black",
                 point_size=20,
                 time_vel=1,
                 border=False,
                 fargs=None,
                 on_step_callback=None):
        #Callbacks and arguments
        self.on_step_callback = on_step_callback
        self.fargs = fargs
        
        #State variables
        self.is_running = False  
        self.current_frame =init_time
        self.time_vel=time_vel
        
        #Matplotlib actors and its inititial configuration    
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.set_grid(grid,border)
        self.ax.tick_params(axis='both', which='major', labelsize=6)
        self.draw(plot_type,data,wall_data,color,point_size)
        self.set_lims(lims[0],lims[1])
        self.set_titles(title,axis_titles)
        self.update_func=update
        
        #Animation
        self.animation=FuncAnimation(self.fig,
                                     self._internal_update,
                                     frames=self._frame_generator,
                                     interval=120,
                                     blit=True, 
                                     cache_frame_data=False,
                                     fargs=(fargs,))
    def _frame_generator(self):
        
        """Main frame generator for matplotlib function animation"""
        
        while True:
            yield self.current_frame
            if self.is_running: #Running callback from SimControl
                self.current_frame += self.time_vel
    def set_titles(self,title,axis_titles):
    
        """Sets a titles of graphic and axis"""
    
        if title!="":
            self.ax.set_title(title,wrap=True)
        if axis_titles!=["",""]:
            self.ax.set_xlabel(axis_titles[0],wrap=True)
            self.ax.set_ylabel(axis_titles[1],wrap=True)
    def _internal_update(self,frame,*args):
        
        """Internal upate logic function."""
        
        #Not running mode
        if not self.is_running:
            return self.line,
        
        #Data extract
        data=self.update_func(frame,*args) if callable(self.update_func) else None
        
        #Graphic drawing
        if data is not None:    
            self.line.set_offsets(np.column_stack([data[0], data[1]])) if hasattr(self.line, 'set_offsets') else self.line.set_data(data[0],data[1])

        #Time callback
        if self.on_step_callback:
            self.on_step_callback(self.current_frame)
        return self.line,
    def set_lims(self,xlim,ylim):     
        
        """Sets graphic limits if wanted"""
        
        if ylim!=None:
            self.ax.set_ylim(ylim[0],ylim[1])
        if xlim!=None:
            self.ax.set_xlim(xlim[0],xlim[1])
    def set_grid(self,grid,border):
        
        """Sets graphic grid and can be choosed between canonical one, custom grid or none grid."""
        
        if isinstance(grid, np.ndarray): #Draws the grid for hiperbolic space 
            self.grid_lines=[]
            for line in grid:
                self.grid_lines.append(self.ax.plot(line.T[0], line.T[1], color="gray"))
            if border!=False: #Draws border
                self.border = self.ax.plot(border[0], border[1], color="black")
        if isinstance(grid, bool): #Draws canonical grid or not depending on if its True or False sentence
            self.ax.grid(grid)
    def draw(self,plot_type,data,wall_data,color,point_size):
    
        """Initial draw of objects like wals or other data."""
    
        if plot_type=="scatter":
            if callable(data[1]):
                self.line = self.ax.scatter(data[0], data[1](data[0]), color=color,s=point_size)
            else:
                self.line = self.ax.scatter(data[0], data[1], color=color,s=point_size)
        elif plot_type=="plot":
            if callable(data[1]):
                self.line, = self.ax.plot(data[0], data[1](data[0]), color=color)
            else:
                self.line, = self.ax.plot(data[0], data[1], color=color)
        if not wall_data is None:
            self.walls=[]
            for wall in wall_data:
                xs,ys=wall.T
                wall,= self.ax.plot(xs,ys, color="black")
                self.walls.append(wall)
    def pack(self):
        """Packing method for viewing"""
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    def set_time(self, new_time):
    
        """Sets custom time and rearrenges all objects in the drawing"""
        
        self.current_frame = new_time
        args = (self.fargs,) if not isinstance(self.fargs, tuple) else self.fargs

        if callable(self.update_func):
            data = self.update_func(new_time, *args) #Gets the actualized data
            
            if data is not None: #Applies the data to objects
                if hasattr(self.line, 'set_offsets'):
                    self.line.set_offsets(np.column_stack([data[0], data[1]]))
                else:
                    self.line.set_data(data[0], data[1])
                self.canvas.draw()
    def set_speed(self,new_time_vel):
        """Change reproduction speed control variable"""
        self.time_vel = new_time_vel       
    def play(self):
        """Sets playing mode to True"""
        self.is_running = True
    def pause(self):
        """Sets playing mode to False"""
        self.is_running = False       
class GraphicDisplay(Graphic):

    """Tkinter display that supports Graphic. Can be closed eliminating them from the root and amplified to ocupy all the root like normal window."""

    def __init__(self,root,
                 data,
                 init_time=0.0,
                 wall_data=None,
                 update=False,
                 lims=[None,None],
                 title="",
                 graph_title="",
                 axis_titles=["",""],
                 grid=False,
                 plot_type="plot",
                 color="black",
                 time_vel=1,
                 border=False,
                 width=0,
                 height=0,
                 on_close_callback=None,
                 on_amplify_callback=None,
                 on_step_callback=None,
                 fargs=None):
        self.frame=tk.Frame(root, width=width, height=height, bd=1, relief="solid")
        
        #Initialize graphic
        super().__init__(self.frame,
                         data,
                         init_time,
                         wall_data,
                         update,
                         lims,
                         graph_title,
                         axis_titles,
                         grid,
                         plot_type=plot_type,
                         color=color,
                         time_vel=time_vel,
                         border=border,
                         fargs=fargs,
                         on_step_callback=on_step_callback)
        
        #Internal State Variables
        self.on_close_callback = on_close_callback
        self.on_amplify_callback = on_amplify_callback
        self.is_amplified = False
        
        #Other internal elements
        self.title_bar=tk.Frame(self.frame, width=width, height=30, bd=1, relief="solid")
        self.title=tk.Label(self.title_bar, text=title, font=("Arial", 10))
        self.close_button=tk.Button(self.title_bar, text="x", font=("Arial", 10),command=self.close)
        self.amplify_button=tk.Button(self.title_bar, text="▢", font=("Arial", 10),command=self.amplify)
    def _internal_pack(self):
        """Packs internal elements in the display"""
        self.title_bar.pack(side="top",fill="both")
        self.close_button.pack(side="right")
        self.amplify_button.pack(side="right")
        self.title.pack(side="left",fill="both")
        super().pack()
    def grid(self,row=0, column=0, padx=0, pady=0, sticky=""):
        """Plots the graphic display in a grid"""
        self.frame.grid(row=row, column=column, padx=padx, pady=pady, sticky=sticky)
        self._internal_pack()
    def close(self):
        """Makes close Callback to tell Simulation Matress pannel that it has been closed"""
        if self.on_close_callback:
            self.on_close_callback(self)
    def amplify(self):
        """Makes amplify Callback to tell Simulation Matress pannel that it has been amplifyed"""
        if self.on_amplify_callback:
            self.on_amplify_callback(self)
    def set_amplify_state(self, is_maximized):
        """Helper to change the icon visually from the Sim Mattress pannel."""
        self.is_amplified = is_maximized
        if is_maximized:
            self.amplify_button.config(text="-") #Minimize simbol
            self.toolbar = NavigationToolbar2Tk(self.canvas,self.frame)
            self.toolbar.update()
        else:
            self.amplify_button.config(text="▢")
class Form():

    """A form is a combination of a text and an entry box or checkbox.You can access the variables in the form via self.get."""

    def __init__(self,root,text,entry_text="",width=0,height=0,fontsize=10,entry_type="numbers"):
        self.frame=tk.Frame(root,width=width,height=height)
        self.label=tk.Label(self.frame, text=text, font=("Arial", fontsize))
        self.entry_type=entry_type
        if entry_type=="numbers":
            self.entry=tk.Entry(self.frame,width=10,justify="right")
            self.entry.insert(0, entry_text)
        elif entry_type=="bool":
            self.check_state=tk.BooleanVar(value=False)
            self.checkbox=tk.Checkbutton(self.frame,width=5,variable=self.check_state)
    def _internal_pack(self):
        """Packs internal elements of the form"""
        self.label.pack(side="left",anchor="w",padx=10)
        if self.entry_type=="numbers":
            self.entry.pack(side="right",anchor="e",padx=10)
        elif self.entry_type=="bool":
            self.checkbox.pack(side="left",anchor="e",padx=10)
    def pack(self,anchor="center",side="left",padx=0,pady=0):
        """Packs elements of the form"""
        self.frame.pack(anchor=anchor,side=side,padx=padx,pady=pady)
        self._internal_pack()
    def place(self,anchor="center",relx=0,rely=0):
        """Places elements of the form"""
        self.frame.place(anchor=anchor,relx=relx,rely=rely)
        self._internal_pack()
    def grid(self,row=0,column=0,padx=0,pady=0,sticky="n"):
        """Sets in grid elements of the form"""
        self.frame.grid(row=row,column=column,padx=padx,pady=pady,sticky=sticky)
        self._internal_pack()
    def grid_remove(self):
        """Removes the widget from the grid but remembers its options."""
        self.frame.grid_remove()
    def get(self):
        """Function that returns the current state of the entry/checkbox of the form"""
        if self.entry_type=="numbers":
            return self.entry.get()
        elif self.entry_type=="bool":
            return self.check_state.get()
#Sub windows
class SimInfo():
    
    """This pannel shows in real time all important and some optional information of the simulation."""

    def __init__(self, root, init_parameters, width=0, height=0):
        self.frame = tk.Frame(root, width=width, height=height, bd=1, relief="solid")
        self.fmt = self._format_value 
        if width > 0 or height > 0:
            self.frame.pack_propagate(False)
            
        #Starts state variables
        self.max_text_width = width - 150
        self.base_value_size = 9
        self.base_value_family = "Arial"
        
        #Title
        self.title_frame = tk.Frame(self.frame)
        self.main_label = tk.Label(self.title_frame, text="Simulation Info", font=("Arial", 14, "bold"))
        self.separator = ttk.Separator(self.frame, orient='horizontal')

        #Data Frame
        self.data_frame = tk.Frame(self.frame)
        label_font = ("Arial", 9, "bold") # Un poco más pequeño para que quepa todo
        value_font = ("Arial", 9)
        
        #Row creator(helper)
        def create_row(row_idx, text):
            tk.Label(self.data_frame, text=text, font=label_font).grid(row=row_idx, column=0, sticky="w", pady=2)
            lbl = tk.Label(self.data_frame, text="---", font=value_font)
            lbl.grid(row=row_idx, column=1, sticky="w", padx=5)
            return lbl

        #Time
        self.lbl_tau = create_row(0, "Time (tau):")
        self.lbl_t   = create_row(1, "Time (t):")
        
        #Visual separator
        ttk.Separator(self.data_frame, orient='horizontal').grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)

        #Velocity
        self.lbl_vel_kasner = create_row(3, "Vel (Scale Fac):") #Kasner velocity
        self.lbl_vel_poinc  = create_row(4, "Vel (Poincaré):")  #Poincare Ball Velocity

        #Position
        self.lbl_pos_beta   = create_row(5, "Pos (Scale Fac):") #Scalae space position
        self.lbl_pos_poinc  = create_row(6, "Pos (Poincaré):")  #Poincare Ball position

        #Static values
        self.lbl_tau.config(text=self.fmt(init_parameters.get('Initial Time', 0)))
    def pack(self):
        """Packs al items"""
        self.frame.pack(side="left", fill="y")
        self.title_frame.pack(side="top", fill="x", pady=(10, 5), padx=5)
        self.main_label.pack(anchor="center")
        self.separator.pack(fill='x', padx=5, pady=5)
        self.data_frame.pack(side="top", fill="both", padx=10, pady=5)
        self.data_frame.columnconfigure(1, weight=1)
    def _set_smart_text(self, label_widget, text):
        
        """Sets the text in the label_widget in the correct size to ocupy the correct amount of screen based on the text. If too much text sets it in the next line"""
        text = str(text)
        if '\x00' in text: text = text.replace('\x00', '')
       
        current_size = self.base_value_size
        family = self.base_value_family 
        f = tkfont.Font(family=family, size=current_size)
        text_width = f.measure(text)
        
        #Reduction of font loop
        while current_size > 7 and text_width > self.max_text_width:
            current_size -= 1
            f.configure(size=current_size)
            text_width = f.measure(text)
        
        #Next line ocuping method
        if text_width > self.max_text_width:
            avg_char_width = f.measure("0")
            chars_per_line = int(self.max_text_width / avg_char_width)
            
            wrapped_lines = textwrap.wrap(text, width=chars_per_line)
            final_text = "\n".join(wrapped_lines)
        else:
            final_text = text
        label_widget.config(text=final_text, font=(family, current_size), justify="left")     
    def _format_value(self, value):
        
        """Puts in the right format numbers or arrays.Uses cientific notation (.3e) just if >= 1000 or < 0.001. Elsewhere uses normal format(.4g)."""
        
        def fmt_item(x):
            """Internal function to have correct format data"""
            try:
                fx = float(x)
                if fx == 0: return "0.0" #Zero format to show that is float
                if abs(fx) >= 1000 or abs(fx) < 0.001: #Condition before going to 4 digits
                    return f"{fx:.2g}"
                else:
                    return f"{fx:.4g}"
            except: #Fallback
                return str(x)

        try:
            if hasattr(value, '__iter__') and not isinstance(value, str): #If array formats all elements
                formatted_items = [fmt_item(x) for x in value]
                return "(" + ", ".join(formatted_items) + ")"
            else: #Is a float or other and formats just the value
                return fmt_item(value)
        except: #Fallback
            return str(value)
    def update_info(self, tau=None, vel_kasner=None, vel_poinc=None, pos_beta=None, pos_poinc=None):
        
        """Function that updates information of simualtion objects with right size in this pannel."""
        
        if tau is not None:
            self._set_smart_text(self.lbl_tau, self.fmt(tau))
            try:
                val_t = np.exp(-float(tau))
                self._set_smart_text(self.lbl_t, self.fmt(val_t))
            except: 
                self.lbl_t.config(text="Inf")

        if vel_kasner is not None: self._set_smart_text(self.lbl_vel_kasner, self.fmt(vel_kasner))
        if vel_poinc is not None: self._set_smart_text(self.lbl_vel_poinc,  self.fmt(vel_poinc))
        
        if pos_beta is not None: self._set_smart_text(self.lbl_pos_beta,   self.fmt(pos_beta))
        if pos_poinc is not None: self._set_smart_text(self.lbl_pos_poinc,  self.fmt(pos_poinc))
class SimConfig():
    
    """Panel that configurates simulation."""
    
    def __init__(self, root, width=0, height=0):
        self.frame = tk.Frame(root, width=width, height=height, bd=1, relief="raised")
        if height > 0:
            self.frame.pack_propagate(False)
        
        #Menu structure
        self.menu_structure = {
            "File": ["Save Simulation", 
                     "Load Simulation"],
            "Edit": ["Add Slice", 
                     "Add Simulation", 
                     "Delete Simulation", 
                     "---", # Visual separators
                     "Modify Simulation conditions", 
                     "Modify Slice conditions", 
                     "---", # Visual separators
                     "Change Position", 
                     "Change Velocity",
                     "Change Matter Content"],
            "Export": ["Export Simulation data (.csv, .txt)", 
                       "Export frame image (.png, .jpg, .bmp)", 
                       "Export video (.mp4)"],
            "Simulate": ["Caos Simulation"],
            "View": ["View SimInfo pannel",
                     "View Spacial curvature",
                     "View particle stela",
                     "View Weyl Chamber zone"]}
        
        #Menu objects
        self.menubuttons = []
        self.separators = []
        self.menus = {} 
        menu_names = list(self.menu_structure.keys())

        #Create all structure from the menu structure from above
        for i, name in enumerate(menu_names):
            mb = tk.Menubutton(self.frame, text=name, font=("Arial", 10), padx=10, pady=5, relief="flat") #Menu buttons
            mb.config(activebackground="#e0e0e0")
            menu = tk.Menu(mb, tearoff=0) #Makes the menubuttons have menu
            for item in self.menu_structure[name]: #Adds internal separators
                if item == "---":
                    menu.add_separator()
                else:
                    menu.add_command(label=item, command=lambda x=item: self._on_menu_click(x))

            mb.config(menu=menu)
            self.menubuttons.append(mb)
            self.menus[name] = menu

            if i < len(menu_names) - 1: #Adds separators 
                sep = ttk.Separator(self.frame, orient='vertical')
                self.separators.append(sep)
    def pack(self):
        """Packs all elements from SimConfig"""
        self.frame.pack(side="top", fill="x")
        for i, mb in enumerate(self.menubuttons):
            mb.pack(side="left", fill="y")
            if i < len(self.separators):
                self.separators[i].pack(side="left", fill="y", padx=2, pady=8)
    def _on_menu_click(self, action_name):
        """Placeholder function to manage clicks"""
        print(f"Selected action: {action_name}")
class SimControl():
    
    """Pannel controls the time flow of simulation"""
    
    def __init__(self,root,init_parameters,width=0,height=0):
        self.frame=tk.Frame(root,width=width,height=height,bd=1, relief="solid")
        self.frame.pack_propagate(False)
        
        #Sets internal state variables
        self.tau_time=init_parameters["Initial Time"]
        self.time_vel=init_parameters["Time Speed"]
        self.stop=True
        
        #Callbacks
        self.callback_on_start_stop = None
        self.callback_on_speed_change = None
        self.callback_set_manual_time= None
        
        #Time control frame and buttons
        self.time_control_frame=tk.Frame(self.frame,width=width*0.5,height=height)
        self.start_stop_btn=tk.Button(self.time_control_frame, text="Start", font=("Arial", 14),command=self.start_stop)
        self.plus_time_btn=tk.Button(self.time_control_frame, text="+ Speed", font=("Arial", 14),command=self.time_vel_up)
        self.less_time_btn=tk.Button(self.time_control_frame, text="- Speed", font=("Arial", 14),command=self.time_vel_down)
        
        #Time control information frame and speed
        self.time_info_frame= tk.Frame(self.frame)
        self.speed_form = Form(self.time_info_frame, text="Speed", entry_text=str(self.time_vel), fontsize=12)
        self.error_label = tk.Label(self.time_info_frame, text="", fg="red", font=("Arial", 10, "bold"))
        self.speed_form.entry.bind("<Return>", self.validate_speed_input)
        self.speed_form.entry.bind("<FocusOut>", self.validate_speed_input)
        
        #Time Variable frame
        self.time_vars=tk.Frame(self.frame,width=width*0.5,height=height)
        self.tau_form = Form(self.time_vars, text="tau=", entry_text=str(self.tau_time), fontsize=12)
        self.tau_form.entry.bind("<Return>", self.validate_time_input)
        self.tau_form.entry.bind("<FocusOut>", self.validate_time_input)
    def _internal_pack(self):
        """Packs all internal objects on SimControl pannel"""
        
        #Tau
        self.time_vars.pack(side="left", fill="y", padx=10)   
        self.time_vars.pack(side="left", padx=10)
        self.tau_form.pack(side="left")
    
        #Information labels and speed
        self.time_info_frame.pack(side="right", padx=20)
        self.error_label.pack(side="left", padx=5)
        self.speed_form.pack()

        #Time Control 
        self.time_control_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.less_time_btn.pack(side="left", padx=5)
        self.start_stop_btn.pack(side="left", padx=5)
        self.plus_time_btn.pack(side="left", padx=5)
    def pack(self):
        """Packs all objects on SimControl pannel"""
        self.frame.pack(side="bottom", fill="x") 
        self._internal_pack()    
    def validate_time_input(self, event=None):
        
        """Tries to save the current value of the manual time input. Returns True if its correct, False if there is an error and its executed by pulsing enter, buttons or clicks out of the box."""
        
        if not self.stop: #Looks first if simulation is stopped as a safety measure
            self.error_label.config(text="You must pause first to change time!") 
            return False
        try: #Tries to set new value, if error triggers error mesage
            new_time = float(self.tau_form.entry.get().strip().replace(',', '.'))
            self.error_label.config(text="")
            if self.callback_set_manual_time: #Makes callback to set new time in all other pannels
                self.callback_set_manual_time(new_time)
            return True
        except ValueError:
            self.error_label.config(text="Time is not in the right float format!")
            return False
    def validate_speed_input(self, event=None):
        
        """Tries to save the current value of the manual speed input. Returns True if its correct, False if there is an error and its executed by pulsing enter, buttons or clicks out of the box."""
        
        try: #Tries to set new value, if error triggers error mesage
            text = self.speed_form.get()
            self.time_vel = float(text)
            self.error_label.config(text="") 
            if self.callback_on_speed_change: #Makes callback to set new time velocity in all other pannels
                self.callback_on_speed_change(self.time_vel)
            return True
        except ValueError:
            self.error_label.config(text="Speed Value mus be float type!") 
            return False    
    def start_stop(self):
        
        """Function that detects changes in start stop button. Changes the state of the simulation running mode by making a callback to other pannels about running changes."""
        
        if self.stop: #Changes mode based on its previous mode
            self.stop = False
            self.start_stop_btn.config(text="Stop")
        else:
            self.stop = True
            self.start_stop_btn.config(text="Start")
        if self.callback_on_start_stop: #Makes the calback to other pannels to ensure the simulation is stopped
            self.callback_on_start_stop(not self.stop)
    def time_vel_up(self):
        """Function asociated with pressing the button to speed up. Changes internal time velocity state variable and makes the calback to update velocities to other pannels"""
        if self.validate_speed_input():
            self.time_vel *= 2
            self.update_speed_form()
        if self.callback_on_speed_change: #Makes the calback to update other pannels
            self.callback_on_speed_change(self.time_vel)
    def time_vel_down(self):
        """Function asociated with pressing the button to speed down. Changes internal time velocity state variable and makes the calback to update velocities to other pannels"""
        if self.validate_speed_input():
            self.time_vel /= 2
            self.update_speed_form()
        if self.callback_on_speed_change:#Makes the calback to update other pannels
            self.callback_on_speed_change(self.time_vel)
    def update_time_displays(self,current_time):
        """Changes time to current_time in the time form"""
        self.time=current_time
        if self.time>1000: #If time is bigger than 1000 is dificult to show so changes to correct format
            self.tau_form.entry.delete(0, tk.END)
            self.tau_form.entry.insert(0, f"{self.time:.4e}")
        else:
            self.tau_form.entry.delete(0, tk.END)
            self.tau_form.entry.insert(0, f"{self.time:.4f}")
    def update_speed_form(self):
        """Sets time velocity text from time velocity form to self.time_vel"""
        self.speed_form.entry.delete(0, tk.END)
        val_to_show = int(self.time_vel) if self.time_vel.is_integer() else self.time_vel
        self.speed_form.entry.insert(0, str(val_to_show))
        self.error_label.config(text="")
class SimulationMattres():

    """Pannel that suports all simulation with all the graphic elements and disposition."""
    
    def __init__(self, root, init_parameters, width=0, height=0):
        self.frame = tk.Frame(root, width=width, height=height, bd=1, relief="solid")
        self.sim_selector = ttk.Notebook(self.frame)
        self.tab = tk.Frame(self.sim_selector)
        self.sim_selector.add(self.tab, text=f"Simulation {1}")
        
        # Internal state variables and Callbacks
        self.time_vel = init_parameters["Time Speed"]
        self.amplified_graphic = None
        self.callback_update_ui = None
        
        # Starts simulation core object
        self.sim = SimulationCore(init_parameters)
        
        #Scrolable area
        container = tk.Frame(self.tab)
        container.pack(fill="both", expand=True)
        
        self.canvas = tk.Canvas(container)
        self.scrollbar = tk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas_frame_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        #Graphic display creation
        self.graphic_displays = []
        margin = 10
        n_per_row = 3

        for col in range(n_per_row):
            self.scrollable_frame.grid_columnconfigure(col, weight=1,uniform="cols")
        gd_width = (width - (2 * margin * (n_per_row + 1))) / n_per_row
        gd_height = 0.5 * height 

        for i in range(self.sim.dim - 1):
            row = i // n_per_row
            col = i % n_per_row
            
            particle_pos, walls, border, grid = self.sim.get_part_pos(i), self.sim.get_walls(i), self.sim.get_border(i), self.sim.get_grid(i)
            slice_def = self.sim.get_slice_def(i)
            
            gd = GraphicDisplay(
                self.scrollable_frame, # Padre correcto
                particle_pos,
                init_time=self.sim.tau,
                wall_data=walls,
                update=self.sim.update,
                plot_type="scatter",
                color="red",
                time_vel=self.time_vel,
                title=f"Slide {i+1}",
                graph_title=f"Base=[{slice_def[0][0]},{slice_def[0][1]}], Offset={slice_def[1]}",
                axis_titles=[f"Direction {slice_def[0][0]}", f"Direction {slice_def[0][1]}"],
                width=gd_width, 
                height=gd_height,
                grid=grid,
                border=[border[:, 0], border[:, 1]],
                fargs=i,
                on_step_callback=self.relay_time_to_ui,
                on_close_callback=self.remove_graphic,
                on_amplify_callback=self.toggle_amplify_graphic)
            self.graphic_displays.append(gd)
            gd.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            self.scrollable_frame.grid_rowconfigure(row, weight=1)
            
            self.redraw_grid()
    def redraw_grid(self):
        """Redraws graphics into the graphic grid."""
        #Sets every graph to non visible
        for gd in self.graphic_displays:
            gd.frame.grid_forget()
            gd.frame.pack_forget()

        #If there is one that is amplified draws just this one
        if self.amplified_graphic:
            self.amplified_graphic.frame.pack(fill="both", expand=True)
            self.canvas.yview_moveto(0)
        else:
            #Elsewhere draws graphic as usual
            n_per_row = 3
            for i, gd in enumerate(self.graphic_displays):
                row = i // n_per_row
                col = i % n_per_row
                gd.grid(row=row, column=col, padx=10, pady=10, sticky="new")
        self.canvas.update_idletasks()
        
        #Creates an event that configures new canvas dimensions and scrollable regions
        event = tk.Event()
        event.width = self.canvas.winfo_width()
        event.height = self.canvas.winfo_height()
        self._on_canvas_configure(event)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))  
    def remove_graphic(self, target_graphic):
        """Function that's called in graphic target_graphic to close it"""
        #Erase graphic from tkinter memory
        target_graphic.frame.destroy()
        
        #Erase graphic from list of graphics
        if target_graphic in self.graphic_displays:
            self.graphic_displays.remove(target_graphic)
        
        #If one graphic was amplified reestarts last postion
        if self.amplified_graphic == target_graphic:
            self.amplified_graphic = None

        #Redraws grid
        self.redraw_grid()
    def toggle_amplify_graphic(self, target_graphic):
        """Function that is called in graphic target_graphic to amplify it"""
        
        if self.amplified_graphic == target_graphic: #If was amplified, unamplifies
            target_graphic.set_amplify_state(False)
            self.amplified_graphic = None
            target_graphic.toolbar.pack_forget()
        else: #Elswhere amplifies target_graphic
            if self.amplified_graphic: #If anotherone was previously amplified unamplifies (for safety)
                self.amplified_graphic.set_amplify_state(False)
            target_graphic.set_amplify_state(True) #Sets graphic correct mode
            self.amplified_graphic = target_graphic
            target_graphic.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        #Redraws all graphics grid
        self.redraw_grid()
    def _on_canvas_configure(self, event):
        """Adjusts width and height of interior canvas frameto evade vertical scroll if not full"""
        canvas_width = event.width
        canvas_height = event.height
        
        if self.amplified_graphic: #Changes the correct size if its amplified one graphic
            self.canvas.itemconfig(self.canvas_frame_id, width=canvas_width, height=canvas_height)
        else: 
            self.canvas.itemconfig(self.canvas_frame_id, width=canvas_width)
            self.scrollable_frame.update_idletasks()
            needed_height = self.scrollable_frame.winfo_reqheight()
            self.canvas.itemconfigure(self.canvas_frame_id, height=needed_height)
    def relay_time_to_ui(self, current_time):
        
        """Function that returns all information data and curring time to all other ui pannels when Callback"""
    
        if self.callback_update_ui:
            p = self.sim.particle
            self.callback_update_ui(tau=current_time,
                                         vel_kasner=p.kasner_vel,
                                         vel_poinc=self.sim.space.vect_scale_to_poinc(p.pos, p.kasner_vel),
                                         pos_beta=p.pos,
                                         pos_poinc=self.sim.space.scale_to_poinc(p.pos))
    def pack(self):
        """Packs all elements in the panel"""
        self.frame.pack(side="right", fill="both", expand=True, padx=0)
        self.sim_selector.pack(fill="both", expand=True)
    def update_simulation_speed(self, new_speed):
        """Sets running state of all graphics to new_speed"""
        self.time_vel = new_speed
        for display in self.graphic_displays:
            display.set_speed(new_speed)
    def toggle_playback(self, running):
        """Sets running state of all graphics. bool. True = Play, False = Pause"""
        for display in self.graphic_displays:
            if running:
                display.play()
            else:
                display.pause()
    def set_manual_time(self, new_time):  
        """Sets time of all graphics and particles to new_time"""
        self.sim.particle.get_position_at_time(new_time, self.sim.walls)
        for display in self.graphic_displays:
            display.set_time(new_time)
#Main windows
class init_UI():

    """Initialization window of the simulation program."""

    def __init__(self,title):
        #Sets window and configurates it
        self.w,self.h=600,400
        
        #Sets internal state variables
        self.closed=False
        self.dilaton_exp_visible = None
        self.coupling_lst_visible = None
        self.pforms_lst_visible = None
        self.form_dilaton_check = None
        self.form_pforms_check = None
        self.parameters={"Initial Kasner Exp":[-1/3,2/3,2/3],
                         "Initial Beta Pos":[1,2,3],
                         "Initial Time":0.0,
                         "Time Speed":1.0,
                         "Dilaton":False,
                         "Kasner Dilaton Exp":0.0,
                         "Coupling Constants":[2,2],
                         "P-Forms":False,
                         "P-Form List":[1,2]}
        self.type_to_functions={"<class 'int'>":    lambda x:int(x),
                                "<class 'float'>":  lambda x:float(eval(str(x))),
                                "<class 'list'>":   lambda x:eval(x),
                                "<class 'complex'>":lambda x:complex(x),
                                "<class 'set'>":    lambda x:set(x),
                                "<class 'bool'>":   lambda x:bool(x),
                                "<class 'str'":     lambda x:str(x)}
        #Creates window
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)
        self.root.geometry(f"{self.w}x{self.h}") 
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        #Text Frame
        txt_frame = tk.Frame(self.root, width=self.w, height=self.h/5)
        txt_frame.pack(side="top", fill="x", pady=10)
        
        #Title
        title = tk.Label(txt_frame, text="Simulation Initialization", font=("Arial", 18))
        title.pack(anchor="center")
        
        #Error text
        self.error_msg=tk.Label(txt_frame, text="", font=("Arial", 8),fg="red")
        self.error_msg.pack()

        #Form Frame
        form_frame = tk.Frame(self.root, width=self.w, height=3*self.h/5)
        form_frame.pack(side="top", fill="both", expand=True, padx=20, pady=5)
        #Configures Form frame disposition
        form_frame.grid_columnconfigure(0, weight=1, uniform="group1")
        form_frame.grid_columnconfigure(1, weight=0) 
        form_frame.grid_columnconfigure(2, weight=1, uniform="group1")
        
        #Form elements
        num_frame=tk.Frame(form_frame, width=self.w/2, height=3*self.h/5)
        num_frame.grid(row=0, column=0, sticky="nsew")
        sep_line = tk.Frame(form_frame, bg="gray",width=1, height=3*self.h/5)
        sep_line.grid(row=0, column=1, sticky="ns", pady=10)
        bool_frame=tk.Frame(form_frame, width=self.w/2, height=3*self.h/5)
        bool_frame.grid(row=0, column=2, sticky="nsew")
        
        #Creates all forms
        self.num_type=[]
        self.bool_type=[]
        keys_to_exclude=["Kasner Dilaton Exp", "Coupling Constants", "P-Form List"]
        parameters_type=[]
        for key in self.parameters.keys():
            parameters_type.append(type(self.parameters[key]))
            if key=="Initial Kasner Exp":
                self.parameters[key]="[-1/3,2/3,2/3]"
            if (key in keys_to_exclude) or isinstance(self.parameters[key], bool):
                self.bool_type.append([key,str(self.parameters[key])])
            else:
                self.num_type.append([key,str(self.parameters[key])])
        self.parameters_type=parameters_type
        self.forms={}
        for i, (key, value) in enumerate(self.num_type):
            num_frame.grid_rowconfigure(i, weight=1)
            
            f = Form(num_frame, "·" + key + " =", value)
            f.grid(row=i, column=0, sticky="ew", padx=10)
            self.forms[key]=f

        bool_frame.grid_rowconfigure(0, weight=0) # El título no necesita estirarse tanto
        matter_title = tk.Label(bool_frame, text="Matter Content:", font=("Arial", 12, "bold"))
        matter_title.grid(row=0, column=0, sticky="w", padx=15 ,pady=(0, 10))
        
        for i,(key, value) in enumerate(self.bool_type):
            row_idx = i + 1
            bool_frame.grid_rowconfigure(row_idx, weight=1)
            if not key in keys_to_exclude:
                f = Form(bool_frame, "-" + key + ":", entry_type="bool")
                f.grid(row=row_idx, column=0, sticky="ew", padx=10)
                f.checkbox.config(command=self.update_dynamic_fields)
            else:
                f = Form(bool_frame, "  ·" + key + ":", value)
            self.forms[key]=f
        
        #Start Button Frame
        btn_frame = tk.Frame(self.root, width=self.w, height=self.h/5)
        btn_frame.pack()
        #Start Button 
        start_button = tk.Button(btn_frame, text="Start", font=("Arial", 14),command=self.form_is_correct)
        start_button.place(relx=0.5, rely=0.5, anchor="center")

        self.root.mainloop()
    def update_dynamic_fields(self):
        """Shows or ocults p-forms forms and dilaton forms."""
        is_dilaton = self.forms["Dilaton"].get()
        is_pforms =self.forms["P-Forms"].get()
        
        if is_dilaton:
            self.forms["Kasner Dilaton Exp"].grid(row=2,column=0, sticky="ew", padx=10) 
        else:
            self.forms["Kasner Dilaton Exp"].grid_remove()
        if is_pforms:
            self.forms["P-Form List"].grid(row=5,column=0, sticky="ew", padx=10) 
        else:
            self.forms["P-Form List"].grid_remove()
        if is_dilaton and is_pforms:
            self.forms["Coupling Constants"].grid(row=3,column=0, sticky="ew", padx=10) 
        else:
            self.forms["Coupling Constants"].grid_remove()
    def extract_parameters(self):
        
        """Returns parameters of the window in its correct format"""
        
        #Processes to right format all elements
        for i, key in enumerate(self.parameters.keys()):
            converter = self.type_to_functions[ str(self.parameters_type[i])]
            self.parameters[key] = converter(self.forms[key].get())

        #Matter integration logic (post-processing)
        if self.parameters["Dilaton"]:
            # if there is matter one has to change kasner velocity[p1, p2, ..., p_phi]
            spatial_kasner = self.parameters["Initial Kasner Exp"]
            phi_exp = self.parameters["Kasner Dilaton Exp"]
            
            #Adds dilaton
            if isinstance(spatial_kasner, list):
                spatial_kasner.append(phi_exp)
            else:
                spatial_kasner = list(spatial_kasner) + [phi_exp]
            #Rewrites original parameter
            self.parameters["Initial Kasner Exp"] = spatial_kasner
    def form_is_correct(self):
    
        """Checks if init form is correct."""
    
        incorrect_message=""
        for i, expected_type in enumerate(self.parameters_type):
            key_name = list(self.parameters.keys())[i]
            param_value = list(self.forms.values())[i].get()
            
            #Looks at the format 
            try:
                type_key = str(expected_type) # "<class 'list'>", etc.
                converter = self.type_to_functions[type_key]
                converted_val = converter(param_value)
                if expected_type == list and not isinstance(converted_val, list): #Extra evaluation for tuples without []
                    converted_val = list(converted_val)
                    
            except Exception:
                incorrect_message = f"Error in '{key_name}': Invalid format for {expected_type.__name__}"
                break #Stops procesing to save computing capacity
        
            #Fisical validations
            try:
                # Kasner exponents and dilaton(i=0)
                if i == 0:
                    k_exp = np.array(converted_val, dtype=np.float64) #Sum and squared sum unity
                    if self.forms["Dilaton"].get(): #Condition if there are dilatons
                        try:
                            p_phi = float(eval(self.forms["Kasner Dilaton Exp"].get()))
                        except ValueError:
                            incorrect_message = "Kasner Dilaton Exp must be a valid number."
                            break
                        if not np.isclose(np.sum(k_exp), 1.0) or not np.isclose(np.sum(k_exp**2)+p_phi**2, 1.0):
                            incorrect_message = "With Dilaton: Sum(p_i)=1 AND Sum(p_i^2) + p_phi^2 = 1"
                    else:
                        if not np.isclose(np.sum(k_exp), 1.0) or not np.isclose(np.sum(k_exp**2), 1.0):
                            incorrect_message = "Kasner exp: Sum must be 1 AND Sum of squares must be 1."

                # Beta Position (i=1)
                elif i == 1:
                    beta = np.array(converted_val, dtype=np.float64)
                    
                    # Timelike condition (in DeWitt metric) Sum(x^2) - (Sum(x))^2 < 0
                    dewitt_norm = np.sum(beta**2) - (np.sum(beta)**2)
                    if not dewitt_norm < 0:
                        incorrect_message = "Initial Pos: Must be Timelike in DeWitt metric."
                    
                    #Weyl Chamber condition b0 < b1 < b2 ...
                    elif np.any(np.diff(beta) <= 0):
                        incorrect_message = "Initial Pos: Betas must be strictly ordered (b0 < b1...)."
                    
                    #Gravitational condition 
                    elif beta[0] <= 0:
                        incorrect_message = "Initial Pos: Smallest beta must be positive (> 0)."

                    #Dimensional consistency with Kasner exponents
                    try:
                        kasner_raw = self.forms["Initial Kasner Exp"].get()
                        kasner_dim = len(list(eval(kasner_raw)))
                        if self.forms["Dilaton"].get():
                            if len(beta) != kasner_dim+1:
                                incorrect_message = f"With Dilaton, Beta Pos must have {kasner_dim + 1} dimensions (Space + Phi)."
                        else:
                            if len(beta) != kasner_dim:
                                incorrect_message = "Dimensions mismatch between Kasner and Beta."
                    except: pass 

                #Initial time (i=2)
                elif i == 2 and converted_val < 0:
                    incorrect_message = "Initial time must be >= 0."

                #Time Speed (i=3)
                elif i == 3 and converted_val < 0:
                    incorrect_message = "Time speed must be >= 0."
                
                #Coupling constants and P-Forms list
                elif i==8 and self.forms["P-Forms"].get():
                    
                    #Condition 0<=p<D
                    kasner_len = len(eval(self.forms["Initial Kasner Exp"].get()))
                    for p_val in converted_val:
                        if not isinstance(p_val, int) or p_val < 0 or p_val >= kasner_len:
                            incorrect_message = f"P-Form rank {p_val} invalid. Must be integer 0 <= p < {kasner_len}."
                    try:#Looks for equal dimension of vectors coupling constant and p-forms list
                        couplings = eval(self.forms["Coupling Constants"].get())
                        if isinstance(couplings, (int, float)): couplings = [couplings] 
                        if len(converted_val) != len(couplings):
                            incorrect_message = "Length of 'P-Form List' must match 'Coupling Constants'."
                    except:
                        incorrect_message = "Invalid format in Coupling Constants List."
            except Exception as e:
                incorrect_message = f"Logic error in '{key_name}': {str(e)}"

            # If error found stops loop to save computational cost
            if incorrect_message != "":
                break

        #If there is no incorrect message destroys innit and gets values. Elsewhere it sets config error mesage
        if incorrect_message == "":
            self.extract_parameters()
            self.root.destroy()
        else:
            self.error_msg.config(text=incorrect_message)
    def on_close(self):
        """Says if the program was closed"""
        self.closed=True
        self.root.destroy()
class UserInterface():
    
    """Principal window of the simulation program."""
    
    def __init__(self,title,init_parameters):
        #Creates root and configurates it.
        self.root=tk.Tk()
        self.root.title(title)
        self.root.bind_all("<Button-1>", lambda event: event.widget.focus_set())
        self.w,self.h = self.root.winfo_screenwidth(),self.root.winfo_screenheight()
        self.root.geometry(f"{self.w}x{self.h}")
        
        #Creates all pannels
        self.config_panel = SimConfig(self.root,width=self.w, height=30)
        self.info_panel = SimInfo(self.root,init_parameters, width=int(0.125*self.w), height=self.h-30)
        self.control_panel =SimControl(self.root,init_parameters,width=int(0.875*self.w), height=int(0.1*self.h))
        self.sim_mattres = SimulationMattres(self.root,init_parameters,width=int(0.875*self.w), height=int(0.9*self.h -10))
        self.pannels=[self.config_panel,self.info_panel,self.control_panel,self.sim_mattres]
        
        #Makes callbacks of diferent pannels
        # Conect Start/Stop from Control
        self.control_panel.callback_on_start_stop = self.sim_mattres.toggle_playback
        # Conect Simulation velocity from Control
        self.control_panel.callback_on_speed_change = self.sim_mattres.update_simulation_speed
        #Conects current Simulation velocity to Control
        self.sim_mattres.callback_update_ui= self._distribute_updates
        #Connects manual time editing on Simulation
        self.control_panel.callback_set_manual_time = self.sim_mattres.set_manual_time
        
        for pannel in self.pannels: #Packs all panels
            pannel.pack()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        #Starts window main loop
        self.root.mainloop()
    def _distribute_updates(self, tau, vel_kasner, vel_poinc, pos_beta, pos_poinc):

        """Updates all simulation information in all pannels"""

        self.info_panel.update_info(
            tau=tau, 
            vel_kasner=vel_kasner, 
            vel_poinc=vel_poinc,
            pos_beta=pos_beta, 
            pos_poinc=pos_poinc)
        self.control_panel.update_time_displays(tau)
    def on_close(self):
        """Manages closing all elements wen window is closed"""
        plt.close('all')
        self.root.destroy()
if __name__ == "__main__":
    
    """Main code"""
    
    init_form = init_UI("Simulation Initialization") #Starts form
    if not init_form.closed: #If form wasn't closed starts simulation
        ui=UserInterface("Cosmological Billard Simulation",init_form.parameters)
#TODO   
    #Clase SimulationCore
        #Error de division de tiempo en update(en x1 actualiza a 1s pero debería tener mas resolución para mayor fluidez)
        #Clase particle:
            #Corregir ciertos choques en 4D y superiores (esquinas)
            #Corregir efecto de "pegado" a los muros
    #clase init_UI  
        #3.Load Simulation
    #clase SimConfig    
        #File:
            #4.Save Simulation
            #5.Load Simulation
        #Edit
            #Add Slice
            #Add Simulation
            #Delete Simulation
            #Modify Simulation conditions
            #Modify Slice contditions
            #Change Position
            #Change Velocity
            #Change Matter Content
        #Export
            #2.Export Simulation data as (.csv,.txt)
            #Export video(.mp4)
        #Simulate
            #Caos Simulation 
        #View
            #View SimInfo pannel
            #View spacial curvature
            #View particle stela
            #View Weyl Chamber zone
    #Clase hyperbolic space
        #Reordenar clases y metodos
            #Sacar vectors_are_base, base_vects de geodesic subspace y ponerlo en Metric vector space
            #Clase  MetricVectorSpace
                #Graham-Schmidt,plane eq to vectors, projections, base, etc.
            #Classe EuclideanSpace,Minkowski,DeWitt
            #Clase euclidean norm como embeding 
                #euclidean norm, inprod,base vect
    #1.Crear estructura de proyecto (init_UI.py,simulator.py,main.py,form1.py,logs,data,etc.)
    #Crear ejecutable .exe para reducir riesgo de error.
