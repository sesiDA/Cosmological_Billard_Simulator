import itertools
import numpy as np
import matplotlib.colors as mcolors
from .math_libs import HyperbolicSpace, DeWittSpace, CoxeterGroup

#WALL CLASS
class Wall(HyperbolicSpace.GeodesicHyperplane):
    """Geodesic hyperplane representing a reflection wall in scale factor space."""
    
    def __init__(self, hyperbolic_space, de_witt_space, func, *args):
        # Calculate geometric properties in Poincaré representation
        vectors, offset = hyperbolic_space.scale_plane_to_poinc_vects(func, *args, dim=de_witt_space.dim)
        if vectors is not None and offset is not None:
            super().__init__(hyperbolic_space, vectors, offset)
            
        self.de_witt_space = de_witt_space
        # Extract normal vector (1-form) and offset in DeWitt space
        self.normal, self.b = de_witt_space.planeq_to_comp(func, *args)
        
        # Classification and Labeling
        self.is_symmetry = (func.__name__ == 'symetry_plane' and len(args) == 2)
        self.indices = args if self.is_symmetry else None
        
        # Generate Label
        try:
            indices_str = str(args).replace(" ", "").replace(",)", ")")
            fname = func.__name__
            if 'symetry' in fname: self.label = f"Sym{indices_str}"
            elif 'grav' in fname: self.label = f"Grav{indices_str}"
            elif 'elec' in fname: self.label = f"Elec{indices_str}"
            elif 'magn' in fname: self.label = f"Magn{indices_str}"
            else: self.label = f"W{indices_str}"
        except: 
            self.label = "Unknown"


#PARTICLE CLASS
class Particle(HyperbolicSpace.Geodesic):
    """Simulates the Universe point moving in the billiard."""
    
    def __init__(self, hyperbolic_space, de_witt_space, init_pos, init_kasner_exp, tau=0.0):
        self.space = hyperbolic_space
        self.de_witt_space = de_witt_space
        
        self.init_pos = np.array(init_pos, dtype=np.float64)
        self.init_vel = np.array(init_kasner_exp, dtype=np.float64)
        
        self.pos = self.init_pos.copy()
        self.kasner_vel = self.init_vel.copy()
        self.tau = tau
        
        # Gauge Fixing: Preserve total energy (sum of velocities) to ensure monotonic collapse
        self.target_lambda = np.sum(self.init_vel)
        if abs(self.target_lambda) < 1e-5:
            self.target_lambda = 1.0 # Fallback for invalid input
            print("Warning: Initial velocity sum is zero. Forcing Lambda=1.0")
            
        self._enforce_kasner_constraints()
    def _enforce_kasner_constraints(self):
        """Re-normalizes velocity to maintain constant expansion rate (Lambda)."""
        current_sum = np.sum(self.kasner_vel)
        if abs(current_sum) < 1e-9:
            # Recovery from numerical collapse
            norm = np.linalg.norm(self.kasner_vel)
            if norm < 1e-9: self.kasner_vel[:] = 1.0 / len(self.kasner_vel)
            else: self.kasner_vel += 1e-2
            current_sum = np.sum(self.kasner_vel)
            
        self.kasner_vel *= (self.target_lambda / current_sum)
    def calculate_current_hopscotch(self, reference_walls):
        """Projects current state onto the Hopscotch diagram (Möbius projection)."""
        # 1. Project to Poincaré Ball
        p_poinc = self.space.scale_to_poinc(self.pos)
        v_poinc = self.space.vect_scale_to_poinc(self.pos, self.kasner_vel)
        
        # 2. Get Geodesic Endpoints (Past/Future in projected space)
        u_plus_vec, u_minus_vec = self.space.get_geodesic_endpoints(p_poinc, v_poinc)
        
        # 3. Define Static Board (Reference Triangle)
        w_A, w_B, w_C = reference_walls[0], reference_walls[1], reference_walls[2]
        scope_normals = [w_A.normal, w_B.normal, w_C.normal]
        
        # Vertices: Inf (A-B), Zero (B-C), MinusOne (A-C)
        v_inf = self.de_witt_space.get_intersection_point(w_A, w_B, scope_normals)
        v_0   = self.de_witt_space.get_intersection_point(w_B, w_C, scope_normals)
        v_m1  = self.de_witt_space.get_intersection_point(w_A, w_C, scope_normals)
        
        # 4. Möbius Transform to U-plane
        u_future = self.space.get_hopscotch_u(u_plus_vec, v_inf, v_0, v_m1)
        u_past   = self.space.get_hopscotch_u(u_minus_vec, v_inf, v_0, v_m1)
        
        return u_past, u_future, w_A.label, w_B.label, w_C.label
    def update(self, tau_target, walls, on_bounce_callback=None):
        """Evolves particle state until tau_target, handling collisions."""
        remaining_tau = tau_target
        tolerance = 1e-12
        max_bounces = 10000 
        
        # Pre-extract wall data for vectorization
        wall_normals = np.array([w.normal for w in walls])
        wall_bs = np.array([w.b for w in walls])
        num_walls = len(walls)

        for _ in range(max_bounces):
            if remaining_tau <= 1e-15: break

            # Calculate collision times
            dists = np.dot(wall_normals, self.pos) - wall_bs.flatten()
            vel_projs = np.dot(wall_normals, self.kasner_vel)
            
            approaching = vel_projs < -1e-15
            
            if not np.any(approaching):
                self.pos += self.kasner_vel * remaining_tau
                self.tau += remaining_tau
                break
                
            t_cols = np.full(num_walls, np.inf)
            
            # Immediate collision check
            close_mask = (dists < tolerance) & approaching
            t_cols[close_mask] = 0.0
            
            # Standard collision time: t = -d / v
            calc_mask = approaching & (~close_mask)
            t_cols[calc_mask] = -dists[calc_mask] / vel_projs[calc_mask]
            t_cols[t_cols < -tolerance] = np.inf # Filter numerical noise
            
            nearest_idx = np.argmin(t_cols)
            nearest_time = t_cols[nearest_idx]
            
            if nearest_time > remaining_tau:
                self.pos += self.kasner_vel * remaining_tau
                self.tau += remaining_tau
                break
            
            # --- COLLISION EVENT ---
            collision_wall = walls[nearest_idx]
            step = max(0.0, nearest_time)
            
            self.pos += self.kasner_vel * step
            self.tau += step
            remaining_tau -= step
            
            self._reflect_vel(collision_wall.normal)
            self._enforce_kasner_constraints() # Restore energy
            
            # Nudge to escape wall
            nudge_dir = self.de_witt_space.raise_index(collision_wall.normal)
            norm_sq = abs(np.dot(nudge_dir, np.dot(self.de_witt_space.g, nudge_dir)))
            if norm_sq > 0:
                self.pos += (nudge_dir / np.sqrt(norm_sq)) * tolerance * 100
            
            if on_bounce_callback:
                on_bounce_callback(collision_wall)
    def _reflect_vel(self, w_form):
        """Reflects velocity vector against a wall normal form."""
        w_vec = self.de_witt_space.raise_index(w_form)
        w_dot_v = np.dot(w_form, self.kasner_vel)
        w_dot_w = np.dot(w_form, w_vec)
        
        if abs(w_dot_w) > 1e-15:
            self.kasner_vel -= (2 * (w_dot_v / w_dot_w)) * w_vec
    def get_position_at_time(self, target_time, walls, on_bounce_callback=None):
        """Simulates from scratch to a specific time (Fast Forward)."""
        self.tau = 0.0
        self.pos = self.init_pos.copy() 
        self.kasner_vel = self.init_vel.copy()
        self._enforce_kasner_constraints()
        
        # Sub-stepping for stability
        steps = np.linspace(0, target_time, 10) 
        for t in steps: 
            self.update(t - self.tau, walls, on_bounce_callback)
            
        if self.tau < target_time: 
            self.update(target_time - self.tau, walls, on_bounce_callback)
            
        return self.pos
    @property
    def current_lambda(self): return np.sum(self.kasner_vel)   
    def get_physical_kasner_exponents(self): return self.kasner_vel / self.current_lambda

#SIMULATION CORE
class SimulationCore():
    """Main simulation engine handling physics, geometry, and logs."""
    
    def __init__(self, init_parameters):
        # Space Initialization
        is_dilaton = init_parameters.get("Dilaton", False)
        n_dil = 1 if is_dilaton else 0
        
        self.beta_dim = len(init_parameters["Initial Kasner Exp"]) 
        self.spatial_dim = self.beta_dim - n_dil
        self.dim = self.beta_dim - 1
        
        self.de_witt_space = DeWittSpace(self.spatial_dim, n_dil)
        
        # 2. Inicializar Espacio Visual
        self.space = HyperbolicSpace(self.dim)

        # 3. CRÍTICO: Conectar la Transformación Métrica
        # Calculamos la matriz T que diagonaliza la métrica G actual
        T_matrix = self.de_witt_space.get_minkowski_transform()
        
        # Se la pasamos al espacio visual. 
        # A partir de ahora, scale_to_poinc usará ESTA matriz.
        self.space.custom_minkowski_transform = T_matrix
        # State Variables
        self.tau = init_parameters["Initial Time"]
        self.t = np.exp(-self.tau) # Belinski-Henneaux time relation
        self.normal_space_pos = np.sqrt(np.exp(-2 * np.array(init_parameters["Initial Beta Pos"])))
        self.current_permutation = np.arange(self.beta_dim)
        
        self.particle = Particle(self.space, self.de_witt_space, 
                                 init_parameters["Initial Beta Pos"], 
                                 init_parameters["Initial Kasner Exp"], 
                                 self.tau)
        
        self.logs = [[self.particle.pos, self.normal_space_pos, self.particle.kasner_vel, self.tau, self.t]]

        # Setup Geometry
        self._setup_walls(init_parameters)
        self._setup_analysis()
        self._setup_slices()
        
        # Initial Computation
        self.set_time(self.tau)

    #SETTERS(Initialization logic separated)
    def _setup_walls(self, params):
        """Generates all reflection walls based on model parameters."""
        raw_walls = []
        grav_indices = range(self.spatial_dim)
        # 1. Symmetry Walls
        if not params["Homogeneous Model"]:
            if self.beta_dim <= 4:
                for a, b in itertools.combinations(grav_indices, 2):
                    raw_walls.append(Wall(self.space, self.de_witt_space, self.symetry_plane, a, b))
            else:
                # Optimización Weyl (Solo raíces simples)
                for i in range(self.beta_dim - 1):
                    raw_walls.append(Wall(self.space, self.de_witt_space, self.symetry_plane, i, i+1))

        # 2. Gravitational Walls
        for a in grav_indices:
            others = [x for x in grav_indices if x != a]
            if len(others) >= 2:
                for b, c in itertools.combinations(others, 2):
                    raw_walls.append(Wall(self.space, self.de_witt_space, self.grav_plane, a, b, c))

        # 3. P-Form Walls
        if params.get("P-Forms", False):
            p_list = params.get("P-Form List", [])
            if params.get("Dilaton", False):
                couplings = params.get("Coupling Constants", [])
            else:
                couplings=np.zeros(len( params.get("Coupling Constants", [])))
            for p, coupling in zip(p_list, couplings):
                # Electric
                for indices in itertools.combinations(range(self.spatial_dim), p):
                    raw_walls.append(Wall(self.space, self.de_witt_space, self.p_form_elec_plane, indices, coupling))
                # Magnetic
                p_magn = self.spatial_dim - p
                if p_magn > 0:
                    for indices in itertools.combinations(range(self.spatial_dim), p_magn):
                        raw_walls.append(Wall(self.space, self.de_witt_space, self.p_form_magn_plane, indices, coupling))

        # 4. Filtering Duplicates
        unique_walls = []
        seen_normals = []
        for w in raw_walls:
            is_dup = False
            for seen in seen_normals:
                if np.allclose(w.normal, seen, atol=1e-9):
                    is_dup = True; break
            if not is_dup:
                unique_walls.append(w); seen_normals.append(w.normal)
        
        all_unique = np.array(unique_walls)
        
        # Aquí se filtran los dominantes (los azules/rojos sólidos)
        self.walls = self.de_witt_space.filter_subdominant_walls(unique_walls)
        
        # 5. Reference Walls (for Hopscotch)
        self.reference_grav_walls = []
        for a in range(self.beta_dim):
            others = [x for x in grav_indices if x != a]
            if len(others) >= 2:
                self.reference_grav_walls.append(Wall(self.space, self.de_witt_space, self.grav_plane, a, others[0], others[1]))

        # 6. Ghost Walls (Visualization)
        self.ghost_walls = []
        
        # Ajusta la dimensión según necesites (3 o 4)
        if self.beta_dim <= 3: 
            
            # A. Preparamos conjunto de identidad para descarte rápido
            dominant_set = set(self.walls)
            
            # B. Pre-calculamos direcciones normalizadas de los DOMINANTES
            #    para compararlas geométricamente con los fantasmas.
            dom_dirs = []
            for d in self.walls:
                n = d.normal
                nm = np.linalg.norm(n)
                if nm > 1e-12: dom_dirs.append(n / nm)

            # C. Iteramos sobre TODOS los muros generados (all_unique)
            #    Nota: 'all_unique' debe ser la lista que creaste en el paso 4 antes del filtrado LP
            #    Si usaste mi código anterior, usa 'consolidated_walls' o regenera 'all_unique' sin filtrar.
            #    Lo ideal es iterar sobre 'all_unique' (la lista cruda filtrada solo por duplicados exactos).
            
            for w in all_unique: 
                # 1. Si el muro YA es dominante, no es un fantasma.
                if w in dominant_set: continue

                # 2. Filtro de Tipo: Solo nos interesan fantasmas Estructurales.
                #    (Descartamos ruido de p-formas subdominantes)
                if not (w.is_symmetry or 'Grav' in w.label): continue

                # 3. LÓGICA DE SOMBREADO (La que pediste)
                is_shadowed = False
                
                # "Descarte el muro si es de simetría y su normal es equivalente a un dominante"
                w_n = w.normal
                w_norm = np.linalg.norm(w_n)
                if w_norm > 1e-12:
                    w_dir = w_n / w_norm
                        
                        # Chequeamos si es paralelo a ALGÚN dominante
                        # abs(dot) ~ 1.0 implica vectores paralelos (mismo sentido o inverso)
                    for d_dir in dom_dirs:
                        if abs(np.dot(w_dir, d_dir)) > 1.0 - 1e-4:
                            is_shadowed = True
                            break
                
                # 4. Si no está "tapado" por un dominante, lo añadimos
                if not is_shadowed:
                    self.ghost_walls.append(w)

        self.ghost_walls = np.array(self.ghost_walls)
        
        # Precompute colors
        if len(self.walls) > 0:
            self.wall_colors = [mcolors.hsv_to_rgb([i/len(self.walls), 0.8, 0.75]) for i in range(len(self.walls))]
        else:
            self.wall_colors = []
            
        self.ghost_colors = [mcolors.hsv_to_rgb([0.0, 0.0, 0.8]) for _ in range(len(self.ghost_walls))]
    def _setup_analysis(self):
        """Initializes group theory analysis tools."""
        self.coxeter_group = CoxeterGroup(self.walls, self.de_witt_space)
        self.coxeter_group_params = self.coxeter_group.get_data_dict()
        
        # --- CORRECCIÓN MASKING: CÁLCULO DEL TESTIGO ESTÁTICO ---
        # Calculamos un punto que sabemos matemáticamente que está dentro de la cámara.
        math_center = self.coxeter_group.find_fundamental_chamber_center()
        
        if math_center is not None:
            self.static_witness = math_center
        else:
            # Fallback: Usamos la posición inicial de la partícula (que por definición empieza dentro)
            print("Warning: Could not compute geometric center. Using initial particle pos.")
            self.static_witness = self.particle.init_pos
        # --- DIAGNÓSTICO TEMPORAL (COPIAR Y PEGAR ESTO) ---
        print("\n--- DIAGNÓSTICO DEL BILLAR ---")
        print(f"Dimensiones: Spatial={self.spatial_dim}, Dilatons={self.de_witt_space.n_dilatons}")
        
        # 1. Verificar la Métrica Inversa (G_inv)
        # Debería tener estructura de bloques. La parte espacial debe tener signos negativos fuera de la diagonal (o estructura Lorentziana implícita).
        print("\n1. Métrica Inversa (DeWitt G^-1):")
        print(np.round(self.de_witt_space.g_inv, 2))
        
        # 2. Verificar Normales de los Muros
        print("\n2. Muros Dominantes Generados:")
        norm_vectors = []
        for i, w in enumerate(self.walls):
            # Normalizamos para ver mejor la estructura
            n = w.normal
            norm_sq = w.normal @ self.de_witt_space.g_inv @ w.normal
            print(f"  Wall {i} [{w.label}]: {np.round(w.normal, 2)} | Norm^2 (Mink): {norm_sq:.2f}")
            norm_vectors.append(w.normal)
            
        # 3. Matriz de Cartan
        print("\n3. Matriz de Cartan Calculada:")
        print(self.coxeter_group.cartan_matrix)
        
        # 4. Autovalores de Cartan (Finitud)
        print("\n4. Autovalores de Cartan:")
        print(np.round(self.coxeter_group.eigenvalues, 4))
        print(f"  Determinante: {self.coxeter_group.determinant:.4f}")
        print(f"  Es Lorentziana (1 autovalor < 0): {self.coxeter_group.is_lorentzian}")
        print(f"  Volumen Finito: {self.coxeter_group.volume_finity}")
        print("--------------------------------\n")
    def _setup_slices(self):
        self.slices = []
        phys_transform = self.de_witt_space.get_minkowski_transform()

        # --- 1. Slice Dinámico (Sin cambios) ---
        if self.beta_dim > 3:
            dyn_slice = self.space.create_dynamic_slice(
                self.particle.pos, 
                self.particle.kasner_vel,
                view_hint_dewitt=self.static_witness 
            )
            self.slices.append(dyn_slice)
        else:
            self.slices.append(self.space.GeodesicHyperplane(
                self.space, [self.space.base[0], self.space.base[1]], np.zeros(self.dim)))
            
        # --- 2. Slices de Inspección (MEJORADO) ---
        corners = self.coxeter_group.find_all_corners()
        
        target_vertex = None
        
        if corners:
            # --- SELECCIÓN INTELIGENTE DEL VÉRTICE ---
            print("\n--- Configurando Slice de Inspección ---")
            
            # A. Obtener dirección de la partícula (Velocidad Kasner)
            # Usamos la velocidad actual (que es la inicial al momento del setup)
            p_vel = self.particle.kasner_vel
            # Normalizamos para comparar puramente dirección (coseno director)
            norm_vel = np.linalg.norm(p_vel)
            p_vel_dir = p_vel / (norm_vel + 1e-15)
            
            # B. Filtrar candidatos: Prioridad a esquinas en el Infinito
            infinite_corners = [c for c in corners if c["is_infinity"]]
            
            # Si existen esquinas infinitas, elegimos entre ellas.
            # Si es un billar finito puro (sin cusps, raro en cosmo), usamos todos los vértices.
            candidates = infinite_corners if infinite_corners else corners
            
            # C. Buscar el candidato con mayor alineación (Max Dot Product)
            best_corner = None
            best_affinity = -np.inf
            
            for c in candidates:
                v_corner = c["vertex_vector"] # Math libs ya lo entrega normalizado
                
                # Afinidad = Cos(theta) en el espacio de parámetros
                # Un valor cercano a 1.0 significa que la partícula va directo a esa esquina
                affinity = np.dot(p_vel_dir, v_corner)
                
                if affinity > best_affinity:
                    best_affinity = affinity
                    best_corner = c
            
            # D. Asignar resultado
            if best_corner:
                target_vertex = best_corner["vertex_vector"]
                
                # Info de depuración útil
                c_type = "Infinito (Cusp)" if best_corner["is_infinity"] else "Finito"
                print(f" -> Objetivo seleccionado: {best_corner['description']}")
                print(f" -> Tipo: {c_type}")
                print(f" -> Afinidad direccional: {best_affinity:.4f}")
            else:
                # Fallback seguro (no debería ocurrir si 'corners' no está vacío)
                target_vertex = corners[0]["vertex_vector"]

        else:
            # Fallback para geometrías sin esquinas (ej. esféricas puras)
            print("Warning: No corners found. Aligning to Chamber Center.")
            target_vertex = self.static_witness

        # 3. Crear el Slice alineado
        # Usa 'create_vertex_aligned_slice' que ya maneja la proyección correcta
        insp_slice = self.space.create_vertex_aligned_slice(
            vertex_vector=target_vertex,
            beta_dim=self.beta_dim,
            metric_transform=phys_transform,
            view_hint_dewitt=self.static_witness,   # Mantiene el 'arriba' coherente
            particle_pos_dewitt=self.particle.pos   # Ancla el slice a la partícula
        )
        
        insp_slice.target_vertex_vector = target_vertex 
        insp_slice.label = "Vertex Perspective"
        self.slices.append(insp_slice)
    #MATH DEFINITIONS (Belinski-Henneaux)
    def symetry_plane(self, betas, a, b):
        return betas[b] - betas[a]
    def grav_plane(self, betas, a, b, c):
        """Gravitational Wall: 2*beta_a + sum_{others} beta_e"""
        val = 2 * betas[a]
        for e in range(len(betas)):
            if e != a and e != b and e != c: val += betas[e] 
        return val
    def p_form_elec_plane(self, betas, indices, coupling=0.0):
        val = sum(betas[i] for i in indices)
        if abs(coupling) > 1e-12: val -= 0.5 * coupling * betas[-1]
        return val   
    def p_form_magn_plane(self, betas, indices, coupling=0.0):
        val = sum(betas[i] for i in indices)
        if abs(coupling) > 1e-12: val += 0.5 * coupling * betas[-1]
        return val

    #GEOMETRIC WRAPPER
    def get_gravitational_indices(self):
        """Identifies which walls form the active triangle for Hopscotch."""
        refs = self.reference_grav_walls
        wall_normals = np.array([w.normal for w in refs])
        wall_bs = np.array([w.b for w in refs])
        
        return self.de_witt_space.find_crossing_sequence(
            self.particle.pos, 
            self.particle.kasner_vel, 
            wall_normals, 
            wall_bs
        )
    
    #HELPERS
    def _fit_boundary_function(self, pts):
        """
        Helper que, dado un conjunto de puntos 2D, devuelve una función lambda f(u,v)
        que representa la ecuación implícita de la curva (Recta o Círculo).
        """
        pts = np.array(pts)
        if len(pts) < 2: return None
        
        # Centrar puntos para estabilidad numérica
        center_mass = np.mean(pts, axis=0)
        pts_c = pts - center_mass
        
        # 1. ¿Es una RECTA? (Colinealidad)
        # Usamos SVD o PCA básico: si el menor valor singular es muy pequeño, es recta.
        # O simplemente comprobamos si el determinante de vectores es cero.
        # Método rápido: vector director
        vec = pts[-1] - pts[0]
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        normal = np.array([-vec[1], vec[0]]) # Normal 2D
        
        # Distancia de puntos a la recta definida por el primero y último
        # d = dot(p - p0, normal)
        dists = np.dot(pts - pts[0], normal)
        is_line = np.all(np.abs(dists) < 1e-4) # Umbral de tolerancia
        
        if is_line:
            # Ecuación Recta: nx*u + ny*v - d = 0
            # Usamos la normal calculada y un punto (p0)
            p0 = pts[0]
            d = np.dot(normal, p0)
            return lambda u, v: normal[0]*u + normal[1]*v - d
            
        else:
            # 2. Es un CÍRCULO (Arco)
            # Ajuste algebraico de círculo: A(x^2+y^2) + Bx + Cy + D = 0
            # Como sabemos que es un muro físico, intentamos ajuste de mínimos cuadrados
            # x^2 + y^2 + Dx + Ey + F = 0  (Asumiendo A=1, círculos normales)
            
            # Sistema: [x_i, y_i, 1] @ [D, E, F].T = - (x_i^2 + y_i^2)
            A_mat = np.column_stack((pts[:, 0], pts[:, 1], np.ones(len(pts))))
            b_vec = -(pts[:, 0]**2 + pts[:, 1]**2)
            
            try:
                # Solución mínimos cuadrados
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                D, E, F = sol
                
                # Retorna función implícita: x^2 + y^2 + Dx + Ey + F
                return lambda u, v: (u**2 + v**2) + D*u + E*v + F
            except:
                return None
    
    #DYNAMICS & EVENTS
    def update(self, target_time_tau, n_slice):
        prev_tau = self.tau
        # Advance Physics
        self.particle.update(target_time_tau, self.walls, on_bounce_callback=self.on_bounce)
        
        # Advance Time
        delta_tau = target_time_tau - prev_tau
        current_lambda = self.particle.current_lambda
        self.t = self.t * np.exp(-current_lambda * delta_tau)
        self.tau = target_time_tau

        # Logging (Physical Scale Factors)
        physical_p = self.particle.get_physical_kasner_exponents()
        raw_space_pos = np.exp(-self.particle.pos) # a_i = exp(-beta_i)
        
        permuted_pos = np.zeros_like(raw_space_pos)
        permuted_pos[self.current_permutation] = raw_space_pos
        
        self.logs.append([self.particle.pos, permuted_pos, physical_p, self.tau, self.t])
        
        return self._get_slice_data(n_slice)
    def fast_update(self, target_time, n_slice):
        return self._get_slice_data(n_slice)
    def on_bounce(self, collision_wall):
        if self.beta_dim > 3: 
             self.slices[0] = self.space.create_dynamic_slice(self.particle.pos, self.particle.kasner_vel)
        
        if collision_wall.is_symmetry:
            a, b = collision_wall.indices
            self.current_permutation[a], self.current_permutation[b] = \
                self.current_permutation[b], self.current_permutation[a]
        
        self._update_hopscotch_state()
    def set_time(self, new_time):
        self.current_permutation = np.arange(self.beta_dim)
        def fast_forward_bounce(collision_wall):
            if collision_wall.is_symmetry:
                a, b = collision_wall.indices
                self.current_permutation[a], self.current_permutation[b] = \
                    self.current_permutation[b], self.current_permutation[a]
        
        self.particle.get_position_at_time(new_time, self.walls, on_bounce_callback=fast_forward_bounce)
        self.tau = new_time
        self.t = np.exp(-self.particle.current_lambda * self.tau)
        
        self.logs = [[self.particle.pos, np.exp(-self.particle.pos), 
                      self.particle.get_physical_kasner_exponents(), self.tau, self.t]]
        
        if self.beta_dim > 3 and self.slices: self.slices[0] = self.space.create_dynamic_slice(self.particle.pos, self.particle.kasner_vel)
        self._update_hopscotch_state()
    def _update_hopscotch_state(self):
        u_minus, u_plus, wa, wb, wc = self.particle.calculate_current_hopscotch(self.reference_grav_walls)
        hop_idx = self.get_gravitational_indices()
        w_from = self.reference_grav_walls[hop_idx['from']].label
        w_to   = self.reference_grav_walls[hop_idx['to']].label
        w_aux  = self.reference_grav_walls[hop_idx['third']].label
        
        self.current_hopscotch_data = {
            "u_minus": u_minus, "u_plus": u_plus,
            "lbl_from": w_from, "lbl_to": w_to, "lbl_aux": w_aux,
            "lbl_a": wa, "lbl_b": wb, "lbl_c": wc
        }

    #GETTERS FOR UI
    def _get_slice_data(self, n_slice):
        """Helper to package data for renderer."""
        data = {
            "particle": self.get_part_pos(n_slice),
            "walls": self.get_walls(n_slice),
            "slice_def": self.get_slice_def(n_slice)
        }
        
        # --- LÓGICA DE EXCLUSIVIDAD ---
        # Determinamos si este slice merece el heatmap.
        # Regla: Si dim > 3, el slice 0 es dinámico (NO mostrar). El slice 1 es inspección (SÍ mostrar).
        # Si dim <= 3, el slice 0 es estático/inspección (SÍ mostrar).
        
        show_heatmap = False
        if n_slice==1:
            show_heatmap = True # Es un inspection slice
        else:
            show_heatmap = False # Es el único slice y es estático
        if show_heatmap:
            # Cacheamos para no recalcular en cada frame
            if not hasattr(self, 'cached_prob_maps'): self.cached_prob_maps = {}
            
            if n_slice not in self.cached_prob_maps:
                # Calculamos solo la primera vez
                self.cached_prob_maps[n_slice] = self.get_fundamental_probability_dist(n_slice)
            data["prob_dist"] = self.cached_prob_maps[n_slice]
        
        return data
    def get_part_pos(self, n_slice):
        return self.slices[n_slice].project_to_hyperplane([self.space.scale_to_poinc(self.particle.pos)])[0]
    def get_walls(self, n_slice):
        projected_walls = []
        if n_slice >= len(self.slices): return []
        current_slice = self.slices[n_slice]
        whitelist = getattr(current_slice, 'relevant_walls', None)
        
        # Iterate over real walls
        for i, wall in enumerate(self.walls): 
            if whitelist is not None and i not in whitelist: continue
            points = current_slice.get_intesect_points(wall)
            if points is not None:
                # Transpose points back for physics_libs/UI compatibility if needed
                # But here we pass points directly to Graphic which expects (N, 2)
                # math_libs.get_intersect returns (N, 2), slice.project returns (N, 2)
                # Wait! Physics_libs previously called .T on border.
                # Let's check Graphic._update_billiard: it expects particle_pos as [x, y] or list.
                # Graphic expects dict with 'data': points array (N, 2) usually.
                projected_walls.append({
                    'data': current_slice.project_to_hyperplane(points),
                    'color': self.wall_colors[i], 
                    'label': wall.label, 
                    'id': i
                })
        
        # Iterate over ghost walls
        if whitelist is None:
            for i, wall in enumerate(self.ghost_walls):
                points = current_slice.get_intesect_points(wall)
                if points is not None:
                    projected_walls.append({
                        'data': current_slice.project_to_hyperplane(points),
                        'color': self.ghost_colors[i], 
                        'label': wall.label, 
                        'id': -1 
                    })
        return projected_walls
    def get_border(self, n_slice):
        return self.slices[n_slice].get_border()
    def get_grid(self, n_slice):
        # Grid lines are list of arrays
        grid_lines = self.slices[n_slice].get_grid()
        projected_grid_lines = []
        for line in grid_lines: 
            projected_grid_lines.append(self.slices[n_slice].project_to_hyperplane(line)) 
        return np.array(projected_grid_lines)
    def get_slice_def(self, n_slice):
        return self.slices[n_slice].base, self.slices[n_slice].offset
    def get_history_data(self, *args):
        if not self.logs: return np.array([]), np.array([])
        times = [entry[4] for entry in self.logs] # t (physical)
        positions = [entry[1] for entry in self.logs] # Scale factors
        return np.array(times), np.vstack(positions)
    def get_hopscotch_frame_data(self, tau, *args):
        return self.current_hopscotch_data
    def get_fundamental_probability_dist(self, n_slice, resolution=500):
        """
        Genera el mapa de probabilidad para el Disco de Poincaré enmascarado por el dominio fundamental.
        Usa ajuste geométrico dinámico para detectar la forma de los muros proyectados.
        """
        # 1. Definir límites del DISCO DE POINCARÉ (Unit Ball)
        # Un poco más de margen para asegurar que cubrimos todo antes de recortar 
        u = np.linspace(-1.0, 1.0, resolution)
        v = np.linspace(-1.0, 1.0, resolution)
        U, V = np.meshgrid(u, v)
        
        # Radio al cuadrado para cálculos
        R2 = U**2 + V**2

        # 2. Calcular Densidad Probabilidad en el DISCO
        # Medida Invariante: P ~ 1 / (1 - r^2)^2
        # Evitamos división por cero y números complejos con un pequeño epsilon y clip
        denom = (1.0 - R2)
        denom = np.where(denom < 1e-6, 1e-6, denom) # Evitar singularidad en r=1
        
        Z_raw = 4.0 / (denom**2)
        
        # Opcional: Clipear suavemente para evitar valores extremos numéricos
        # (Aunque con logaritmo es menos necesario, es buena práctica)
        # Z estará típicamente entre 1.3 (centro) y 15 (borde muy cercano)
        Z = np.clip(Z_raw, 0, 15.0)
        
        # MÁSCARA BASE: El Disco Unitario
        # Nada existe fuera de r=1 en este modelo
        valid_mask = (R2 < 0.999) 

        # 3. Obtener el Testigo (Partícula)
        # Proyectamos la partícula al mismo plano (Disco)
        part_raw = self.slices[n_slice].project_to_hyperplane([self.space.scale_to_poinc(self.particle.pos)])
        
        if len(part_raw) == 0: return {"U": U, "V": V, "Z": np.zeros_like(Z)}
        
        # Coordenadas (u, v) de la partícula en el plano 2D
        p_u, p_v = part_raw[0][0], part_raw[0][1]

        # 4. Enmascarar Muros
        for wall in self.walls:
            pts_3d = self.slices[n_slice].get_intesect_points(wall)
            if pts_3d is None: continue
            
            pts_2d = self.slices[n_slice].project_to_hyperplane(pts_3d)
            if len(pts_2d) < 2: continue 
            
            func_wall = self._fit_boundary_function(pts_2d)
            
            if func_wall:
                val_grid = func_wall(U, V)
                
                # Evaluamos la función de frontera en la POSICIÓN DE LA PARTÍCULA
                val_part = func_wall(p_u, p_v)
                
                # Si el grid tiene el mismo signo que la partícula, es zona válida.
                mask_wall = (np.signbit(val_grid) == np.signbit(val_part))
                valid_mask &= mask_wall

        Z_masked = np.where(valid_mask, Z, np.nan)
        return {"U": U, "V": V, "Z": Z_masked}