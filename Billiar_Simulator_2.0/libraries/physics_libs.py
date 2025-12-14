import itertools
import numpy as np
from .math_libs import HyperbolicSpace

class Wall(HyperbolicSpace.GeodesicHyperplane):
    
    """Geodesic hyperplane in the scale factor representation"""
    
    def __init__(self,space,func,*args):
        self.vectors,self.offset=space.scale_plane_to_poinc_vects(func,*args,dim=space.dim+1)
        if not self.vectors is None and not self.offset is None:
            super().__init__(space,self.vectors,self.offset)
        self.normal,self.b=self.space.planeq_to_comp(func,*args,dim=space.dim+1)   
class Particle():

    """Simulates a particle that represents the expansion of the casner axes."""

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
        max_bounces = 10000 #Max bounces to not explode in corners
       
        for _ in range(max_bounces):
            if remaining_tau <= 1e-15:
                break

            nearest_collision_time = float('inf')
            collision_wall = None

            #Detects colisions
            for wall in walls:
                #Sets distance and velocity to wall 
                dist = np.dot(wall.normal, self.pos) - wall.b
                vel_proj = np.dot(wall.normal, self.kasner_vel)

                if vel_proj < 0.0: #If projected velocity is small is near to wall 
                    
                    if dist<tolerance:#Checks for numerical errors if distance is small
                        t_col=0.0
                    else:
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
            temp_vel=self.kasner_vel
            self.kasner_vel = temp_vel - (2*(w_dot_v / w_dot_w))*w_vec
    def get_position_at_time(self, target_time, walls):
        
        """Estimates the exact position in arbitrary time."""

        self.tau = 0.0
        self.pos = self.init_pos.copy() 
        self.kasner_vel = self.init_vel.copy()
        
        time=np.linspace(0,target_time,100)
        for t in time:
            self.update(t, walls) #Updates position at this time
        
        return self.pos
    @property
    def current_lambda(self):
        """Devuelve la suma de las velocidades actuales: Lambda = sum(v_k)"""
        return np.sum(self.kasner_vel)
    def get_physical_kasner_exponents(self):
        """Devuelve p_i = v_i / Lambda"""
        return self.kasner_vel / self.current_lambda
#Simulation Core
class SimulationCore():
    
    """Simulation Object. Contains all planes and particles that can be added aswell as the space. Has update function to update its informatio in time and can get certain information from it to make graphics like border and grid of slices,particle position, etc."""
    
    def __init__(self,init_parameters):

        self.beta_dim=len(init_parameters["Initial Kasner Exp"])
        self.dim=self.beta_dim-1
        
        self.space=HyperbolicSpace(self.dim)
        
        self.tau=init_parameters["Initial Time"]
        self.t=np.e**(-self.tau)
        self.normal_space_pos=np.sqrt(np.e**(-2*np.array(init_parameters["Initial Beta Pos"])))
        
        #Creates the particle and starts it in the initial setted time
        self.particle=Particle(self.space,init_parameters["Initial Beta Pos"],init_parameters["Initial Kasner Exp"],self.tau)
        
        self.logs=[[self.particle.pos,self.normal_space_pos,self.particle.kasner_vel,self.tau,self.t]]
        
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
        if self.beta_dim>=4:
            self.filter_subdominant_walls()
        
        #if init time is not 0.0 makes simulates particle position to that moment but internally
        self.particle.get_position_at_time(self.tau,self.walls) 
    def filter_subdominant_walls(self, tolerance=1e-5):
        """
        Descarta los muros cuyas normales pueden formarse sumando 
        otras normales existentes en la lista.
        """
        dominant_walls = []
        
        # Iteramos sobre cada muro "candidato" a ser eliminado
        for i, candidate in enumerate(self.walls):
            is_subdominant = False
            v_cand = candidate.normal

            # Comparamos contra todas las combinaciones de pares del resto de la lista
            # Buscamos si v_cand = v_a + v_b
            for j, wall_a in enumerate(self.walls):
                for k, wall_b in enumerate(self.walls):
                    
                    # No usamos el muro candidato para descomponerse a sí mismo
                    # (aunque sí permitimos wall_a == wall_b para detectar casos como 2*beta)
                    
                    # Nota: Para ser muy estrictos, un muro no debería eliminarse a sí mismo,
                    # pero en billares BKL, un muro 'w' nunca es igual a 'w + algo_positivo'.
                    # Solo necesitamos evitar la identidad trivial si hubiera vectores nulos (que no debe haber).
                    
                    v_sum = wall_a.normal + wall_b.normal
                    
                    # Chequeo de igualdad con tolerancia (Float safe)
                    if np.allclose(v_cand, v_sum, atol=tolerance):
                        is_subdominant = True
                        # print(f"DEBUG: {candidate.name} es subdominante. Se forma con {wall_a.name} + {wall_b.name}")
                        break
                
                if is_subdominant:
                    break
            
            # Si no encontramos ninguna suma que lo genere, es Dominante (Raíz Simple)
            if not is_subdominant:
                dominant_walls.append(candidate)

        # Actualizamos la lista oficial
        self.walls = dominant_walls
        return self.walls
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
    def update(self, target_time_tau, n_slice):
        """Updating simulation"""
        
        prev_tau = self.tau
        prev_lambda = self.particle.current_lambda # Lambda del vuelo libre anterior
        
        self.particle.update(target_time_tau, self.walls)
        delta_tau = target_time_tau - prev_tau
        
        current_lambda = self.particle.current_lambda
        
        self.t = self.t * np.exp(-current_lambda * delta_tau)
        
        self.tau = target_time_tau

        physical_p = self.particle.get_physical_kasner_exponents()
        
        self.normal_space_pos = np.sqrt(np.e**(-2*self.particle.pos))
        self.logs.append([self.particle.pos, self.normal_space_pos, physical_p, self.tau, self.t])
        
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
            print(wall.normal)
            points = self.slices[n_slice].get_intesect_points(wall)
            if points is  not None:
                projected_walls.append(self.slices[n_slice].project_to_hyperplane(points)) #Projects point into slice
        return projected_walls
    def get_history_data(self,*args):
        data_len = len(self.logs)
        if data_len == 0:
            return np.array([]), np.array([]) 
        times = [entry[4] for entry in self.logs]
        positions = [entry[1] for entry in self.logs]
        return np.array(times), np.vstack(positions)