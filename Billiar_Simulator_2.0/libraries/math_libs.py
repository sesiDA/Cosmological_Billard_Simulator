import math
import numpy as np
from scipy.linalg import null_space

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

