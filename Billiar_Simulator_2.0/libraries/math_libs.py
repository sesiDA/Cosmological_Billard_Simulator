import math
import numpy as np
import itertools
from scipy.linalg import null_space, block_diag
from scipy.optimize import linprog,minimize, NonlinearConstraint
# --- SPACES ---

class MetricVectorSpace:
    """Base class for vector spaces with a metric tensor."""

    def __init__(self, dim, g=None):
        self.dim = dim
        if g is None:
            self.g = np.eye(dim)
        else:
            self.g = np.array(g)
            if self.g.shape != (dim, dim):
                raise ValueError(f"Metric dimension mismatch. Expected ({dim}, {dim})")
        
        try:
            self.g_inv = np.linalg.inv(self.g)
        except np.linalg.LinAlgError:
            print("Warning: Metric singular, using pseudo-inverse.")
            self.g_inv = np.linalg.pinv(self.g)
    def lower_index(self, vector):
        return self.g @ vector
    def raise_index(self, form):
        return self.g_inv @ form
    def inner_product(self, v1, v2, is_v1_form=False, is_v2_form=False):
        if not is_v1_form and not is_v2_form:
            return v1 @ self.g @ v2
        elif is_v1_form != is_v2_form:
            return np.dot(v1, v2)
        else:
            return v1 @ self.g_inv @ v2
    def squared_norm(self, v, is_form=False):
        return self.inner_product(v, v, is_v1_form=is_form, is_v2_form=is_form)
    def norm(self, v, is_form=False):
        return np.sqrt(np.abs(self.squared_norm(v, is_form)))
    def lineal_dep(self, vectors):
        if len(vectors) == 0: return False
        return np.linalg.matrix_rank(vectors) == vectors.shape[0]
    def canonic_base(self, dim=None):
        dim = dim if dim is not None else self.dim
        return np.eye(dim)
    def graham_schmidt(self, vectors):
        """
        Orthonormalizes a set of vectors using Robust QR Decomposition.
        Replaces manual loop for numerical stability and speed.
        """
        vectors = np.array(vectors)
        if vectors.size == 0: return vectors
        
        # QR decomposes A = Q * R. 
        # A needs to be (M, N) where N is number of vectors.
        # So we transpose input (N_vecs, Dim) -> (Dim, N_vecs)
        Q, _ = np.linalg.qr(vectors.T)
        
        # Q columns are orthonormal. We return them as rows.
        # We only take as many as the rank of input to avoid generating null space noise
        rank = np.linalg.matrix_rank(vectors)
        return Q[:, :rank].T

    def complete_orthonormal_basis(self, initial_vectors, target_dim=2):
        """
        Robust implementation that prioritizes directions orthogonal to existing ones.
        """
        basis = list(self.graham_schmidt(initial_vectors))
        
        # Si ya terminamos, salir
        if len(basis) >= target_dim:
            return np.array(basis[:target_dim])

        # Estrategia de relleno: Usar la base canónica pero mezclada
        # para evitar alineación accidental con ejes singulares.
        candidates = list(np.eye(self.dim))
        
        # Mezclamos candidatos si es necesario para romper simetrías, 
        # o simplemente iteramos intentando añadir ortogonalidad.
        for v in candidates:
            # Proyectar v contra la base actual para ver si aporta algo nuevo
            v_proj = v.copy()
            for b in basis:
                v_proj -= np.dot(v_proj, b) * b # Gram-Schmidt parcial rápido
            
            if np.linalg.norm(v_proj) > 1e-6:
                basis.append(v_proj / np.linalg.norm(v_proj))
            
            if len(basis) >= target_dim: break
        
        # Paso final de limpieza (re-ortonormalizar todo el conjunto)
        return self.graham_schmidt(np.array(basis))[:target_dim]

    def _svd(self, matrix):
        return np.linalg.svd(matrix)

    def normal_vector(self, matrix):
        return self._svd(matrix)[2][-1]

    def planeq_to_comp(self, func, *args, dim=None):
        dim = dim if dim is not None else self.dim
        origin = np.zeros(dim)
        neg_b = func(origin, *args)
        b = -neg_b
        basis = np.eye(dim)
        n = np.zeros(dim)
        for i in range(dim):
            val = func(basis[i], *args)
            n[i] = val - neg_b 
        return n, np.array(b)

class EuclideanSpace(MetricVectorSpace):
    def __init__(self, dim):
        super().__init__(dim, g=None) 

    def euclidean_in_prod(self, p1, p2): return np.dot(p1, p2)
    def euclidean_norm(self, p1): return np.linalg.norm(p1)
    def euclidean_distance(self, p1, p2): return np.linalg.norm(p1 - p2)
    
    def angles(self, p1, p2):
        n1, n2 = self.euclidean_norm(p1), self.euclidean_norm(p2)
        if n1 < 1e-12 or n2 < 1e-12: return 0.0
        cos = np.clip(np.dot(p1, p2) / (n1 * n2), -1.0, 1.0)
        return math.acos(cos)

class DeWittSpace(MetricVectorSpace):
    def __init__(self, spatial_dim, n_dilatons=0):
        self.spatial_dim = spatial_dim
        self.n_dilatons = n_dilatons
        total_dim = spatial_dim + n_dilatons
        
        # --- 1. Parte Gravitacional (Estándar BKL) ---
        # G_ij = delta_ij - 1
        g_grav = np.eye(spatial_dim) - np.ones((spatial_dim, spatial_dim))
        
        # --- 2. Parte Dilatón (Física) ---
        if n_dilatons > 0:
            # Construimos G_dil base (Identidad) escalada por la normalización física
            g_dil = np.eye(n_dilatons)
            
            # Combinamos en la Métrica Covariante G
            self.g = block_diag(g_grav, g_dil)
        else:
            self.g = g_grav
        try:
            g_inv = np.linalg.inv(self.g)
        except np.linalg.LinAlgError:
            print("Warning: Metric singular, using pseudo-inverse.")
            g_inv = np.linalg.pinv(self.g)
        # Inicializamos la clase padre pasando ya la métrica corregida
        super().__init__(dim=total_dim, g=self.g)
        self.g_inv = g_inv
    def get_minkowski_transform(self):
        """
        Calcula la matriz T tal que T.T @ diag(-1, 1, 1...) @ T = G_inv.
        Necesaria para proyectar correctamente desde este espacio métrico híbrido 
        al espacio de Minkowski/Poincaré.
        """
        # Diagonalizamos la métrica G real (autovalores y autovectores)
        vals, vecs = np.linalg.eigh(self.g)
        
        # Ajustamos signo de autovectores para consistencia (opcional pero recomendado)
        # Queremos mapear el autovalor negativo (tiempo) al primer índice
        # Ordenamos: primero el negativo, luego los positivos
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]

        # --- CORRECCIÓN CRÍTICA ---
        # Forzar que el autovector temporal (índice 0) apunte hacia el futuro (suma positiva).
        # Sin esto, 't' puede salir negativo y la proyección (t + rho) deforma la escala.
        if np.sum(vecs[:, 0]) < 0:
            vecs[:, 0] *= -1
        # --------------------------
        
        # La transformación T convierte coordenadas de base métrica a base Minkowski diagonal
        # T = V @ diag(sqrt(|lambda|))
        scale_mat = np.diag(np.sqrt(np.abs(vals)))
        
        # Nota: Retornamos la inversa/transpuesta adecuada para actuar sobre vectores contravariantes
        # Si v_mink = T @ v_dewitt
        return scale_mat @ vecs.T
    def get_intersection_point(self, wall_A, wall_B, scope_normals):
        idx_map = {}
        for i, n in enumerate(scope_normals):
            if np.allclose(n, wall_A.normal, atol=1e-12): idx_map['A'] = i
            elif np.allclose(n, wall_B.normal, atol=1e-12): idx_map['B'] = i
            else: idx_map['C'] = i
        if 'C' not in idx_map: return np.zeros(self.dim)

        n_scope = len(scope_normals)
        gram = np.zeros((n_scope, n_scope))
        for i in range(n_scope):
            for j in range(i, n_scope):
                val = self.inner_product(scope_normals[i], scope_normals[j], True, True)
                gram[i, j] = gram[j, i] = val
        try:
            coeffs = np.linalg.solve(gram, np.eye(n_scope)[idx_map['C']])
        except np.linalg.LinAlgError:
            return np.zeros(self.dim)

        v_int = np.zeros(self.dim)
        for i, n_form in enumerate(scope_normals):
            v_int += coeffs[i] * self.raise_index(n_form)
            
        if v_int[np.argmax(np.abs(v_int))] < 0:
            v_int = -v_int
            
        if np.linalg.norm(v_int) < 1e-12: return np.zeros(self.dim)
        h_space = HyperbolicSpace(self.dim - 1)
        return h_space.scale_to_poinc(v_int)
    
    def filter_subdominant_walls(self, walls, tolerance=1e-5):
        """
        Filtra muros redundantes con PRE-PROCESADO DE COLISIONES.
        
        1. Agrupa muros paralelos (coincidentes) y elige un único 'campeón' por dirección
           (el más restrictivo o el primero en la lista).
        2. Aplica Programación Lineal solo a los campeones para limpiar geometría redundante.
        
        Esto evita que muros idénticos se eliminen mutuamente.
        """
        if not walls: return np.array([])
        
        # --- FASE 1: DEDUPLICACIÓN GEOMÉTRICA (El torneo de campeones) ---
        # Si dos muros son paralelos, solo uno puede sobrevivir antes de ir al LP.
        
        kept_indices = []
        is_active = np.ones(len(walls), dtype=bool)
        
        # Pre-calculamos normas para eficiencia y comparación correcta de 'b'
        norms = []
        scaled_bs = [] # b efectivo (b / ||n||) para comparar cuán restrictivo es el muro
        
        for w in walls:
            n_mag = np.linalg.norm(w.normal)
            if n_mag < 1e-12:
                norms.append(np.zeros_like(w.normal))
                scaled_bs.append(-np.inf) # Muro inválido
            else:
                norms.append(w.normal / n_mag)
                scaled_bs.append(w.b / n_mag)

        for i in range(len(walls)):
            if not is_active[i]: continue
            
            # Comparamos el muro 'i' con todos los siguientes 'j'
            for j in range(i + 1, len(walls)):
                if not is_active[j]: continue
                
                # Chequeo de Paralelismo: Producto punto ~ 1.0
                dot = np.dot(norms[i], norms[j])
                
                if dot > 1.0 - 1e-4: # Son paralelos y apuntan al mismo lado
                    # ¡CONFLICTO! Tenemos dos muros imponiendo la misma dirección.
                    # Debemos quedarnos con el más restrictivo (mayor b efectivo).
                    # Si son iguales, nos quedamos con 'i' (preserva prioridad de entrada).
                    
                    if scaled_bs[j] > scaled_bs[i] + tolerance:
                        # 'j' es más fuerte (está más "adentro"). 'i' es redundante.
                        is_active[i] = False
                        break # 'i' ha muerto, dejamos de compararlo
                    elif scaled_bs[i] > scaled_bs[j] + tolerance:
                        # 'i' es más fuerte. 'j' es redundante.
                        is_active[j] = False
                    else:
                        # Son geométricamente IDÉNTICOS.
                        # Matamos a 'j' para evitar "destrucción mutua" en el LP.
                        # (Mantenemos 'i' porque viene antes en la lista, y asumimos
                        # que la lista viene ordenada por prioridad desde physics_libs).
                        is_active[j] = False
        
        # Recopilamos los sobrevivientes de la Fase 1
        unique_walls = [walls[k] for k in range(len(walls)) if is_active[k]]
        
        # --- FASE 2: FILTRADO MATEMÁTICO (LP) ---
        # Ahora que no hay duplicados exactos, el LP es seguro.
        
        dominant_walls = []
        num_unique = len(unique_walls)
        
        # Preparamos matrices para LP: -n*x <= -b
        if num_unique == 0: return np.array([])
        
        A_all = np.array([-w.normal for w in unique_walls])
        b_all = np.array([-w.b for w in unique_walls])

        for i in range(num_unique):
            candidate = unique_walls[i]
            
            # Construimos restricciones SIN el candidato actual
            mask = np.ones(num_unique, dtype=bool)
            mask[i] = False
            
            A_others = A_all[mask]
            b_others = b_all[mask]
            
            # Si no hay otros muros, este es dominante por defecto
            if len(A_others) == 0:
                dominant_walls.append(candidate)
                continue
            
            # Minimizamos la proyección en la dirección del candidato
            c_obj = candidate.normal
            
            try:
                # bounds=(None, None) es vital para espacio logarítmico (betas pueden ser neg)
                res = linprog(c_obj, A_ub=A_others, b_ub=b_others, bounds=(None, None), method='highs')
                
                if res.success:
                    z_min = res.fun
                    # LÓGICA AFÍN:
                    # Si lo que permiten los OTROS muros (z_min) ya cumple la restricción
                    # de este muro (z >= b), entonces este muro no corta nada nuevo.
                    # Usamos un margen negativo pequeño para robustez numérica.
                    if z_min >= candidate.b - tolerance:
                        pass # Redundante
                    else:
                        dominant_walls.append(candidate)
                else:
                    # Si el problema es ilimitado o falla, asumimos que el muro es necesario
                    dominant_walls.append(candidate)
                    
            except Exception as e:
                # Fallback seguro
                dominant_walls.append(candidate)
                
        return np.array(dominant_walls)

    def find_crossing_sequence(self, pos, vel, wall_normals, wall_offsets):
        dists = np.dot(wall_normals, pos) - wall_offsets
        vel_projs = np.dot(wall_normals, vel)
        with np.errstate(divide='ignore', invalid='ignore'): 
            times = -dists / vel_projs
        valid_past = (times < 1e-9) 
        if np.any(valid_past):
            times_past = times.copy(); times_past[~valid_past] = -np.inf
            idx_from = np.argmax(times_past)
        else: idx_from = np.argmin(np.abs(dists))
        
        valid_future = (times > 1e-9)
        valid_future[idx_from] = False 
        if np.any(valid_future):
            times_fut = times.copy(); times_fut[~valid_future] = np.inf
            idx_to = np.argmin(times_fut)
        else:
            dists_c = np.abs(dists.copy()); dists_c[idx_from] = np.inf
            idx_to = np.argmin(dists_c)
        all_idxs = set(range(len(wall_normals)))
        remaining = list(all_idxs - {idx_from, idx_to})
        idx_aux = remaining[0] if remaining else -1
        return {'from': idx_from, 'to': idx_to, 'third': idx_aux}

class MinkowskiSpace(MetricVectorSpace):
    def __init__(self, dim=2):
        g = np.eye(dim)
        g[0, 0] = -1.0
        super().__init__(dim, g=g)

    def mink_norm2(self, p): return self.squared_norm(p)
    def mink_in_prod(self, p1, p2): return self.inner_product(p1, p2)
    
    def scale_to_mink_matrix(self, dim=None):
        dim = dim if dim is not None else self.dim
        G = np.eye(dim) - np.ones((dim, dim))
        vals, vecs = np.linalg.eigh(G)
        if np.sum(vecs[:, 0]) < 0: vecs[:, 0] *= -1
        scale_mat = np.diag(np.sqrt(np.abs(vals)))
        return scale_mat @ vecs.T

class HyperbolicSpace(MetricVectorSpace):
    def __init__(self, d=2):
        super().__init__(d)
        self.base = np.eye(d)
        self.superspace = MinkowskiSpace(d + 1)
        self.embedding = EuclideanSpace(d)
        self.custom_minkowski_transform = None
    def hyperbolic_distance(self, p1, p2):
        sq_norm_p1 = self.embedding.euclidean_norm(p1)**2
        sq_norm_p2 = self.embedding.euclidean_norm(p2)**2
        dist_sq = self.embedding.euclidean_distance(p1, p2)**2
        delta = 2 * dist_sq / ((1 - sq_norm_p1) * (1 - sq_norm_p2))
        return math.acosh(1 + delta)
    
    def moebius_add(self, p1, p2):
        p1_sq = np.dot(p1, p1)
        p2_sq = np.dot(p2, p2)
        dot = np.dot(p1, p2)
        denom = 1 + p1_sq * p2_sq + 2 * dot
        if abs(denom) < 1e-12: return np.zeros_like(p1)
        num = (1 + 2*dot + p2_sq) * p1 + (1 - p1_sq) * p2
        return num / denom
    def _moebius_cross_ratio(self, z, z_inf, z_0, z_m1):
        if abs(z - z_inf) < 1e-12: return np.inf
        denom = z_m1 - z_0
        if abs(denom) < 1e-12: return np.nan
        K = -1.0 * (z_m1 - z_inf) / denom
        w = K * (z - z_0) / (z - z_inf)
        return w.real

    def scale_to_poinc(self, betas):
        dim = len(betas)
        # USAR TRANSFORMACIÓN CUSTOM SI EXISTE
        if self.custom_minkowski_transform is not None:
            T = self.custom_minkowski_transform
        else:
            T = self.superspace.scale_to_mink_matrix(dim)
            
        betas_mink = T @ betas
        norm_sq = self.superspace.mink_norm2(betas_mink)
        
        if norm_sq < -1e-9:
            rho = np.sqrt(abs(norm_sq))
            denom = betas_mink[0] + rho
            if abs(denom) < 1e-9: return np.zeros(dim - 1)
            return betas_mink[1:] / denom
        elif norm_sq <= 1e-9:
            if abs(betas_mink[0]) < 1e-12: return np.zeros(dim - 1)
            return betas_mink[1:] / betas_mink[0]
        else:
            return np.zeros(dim - 1)
    def vect_scale_to_poinc(self, betas, betas_vel):
        dim = len(betas)
        if self.custom_minkowski_transform is not None:
            T = self.custom_minkowski_transform
        else:
            T = self.superspace.scale_to_mink_matrix(dim)
        Y = T @ betas
        V = T @ betas_vel
        y_norm_sq = self.superspace.mink_norm2(Y)
        if y_norm_sq >= -1e-9: return np.zeros(dim - 1)
        rho = np.sqrt(abs(y_norm_sq))
        rho_dot = -self.superspace.mink_in_prod(Y, V) / rho
        denom = Y[0] + rho
        if abs(denom) < 1e-9: return np.zeros(dim - 1)
        numerator = V[1:] * denom - Y[1:] * (V[0] + rho_dot)
        return numerator / (denom**2)
    def scale_plane_to_poinc_vects(self, func, *args, dim=None):
        n, b = self.planeq_to_comp(func, *args, dim=dim)
        
        # Obtener Transformación Correcta
        if self.custom_minkowski_transform is not None:
            T = self.custom_minkowski_transform
        else:
            T = self.superspace.scale_to_mink_matrix(dim=dim)
            
        T_inv = np.linalg.inv(T)
        n_mink = n @ T_inv
        
        # --- CORRECCIÓN DE ESTABILIDAD ---
        # Separamos parte temporal (índice 0) y espacial
        n_time = n_mink[0]
        n_space_vec = n_mink[1:]
        norm_space = np.linalg.norm(n_space_vec)
        
        # Si el muro es casi nulo (light-like), norm_space ~= abs(n_time).
        # Esto pasa mucho en supergravedad D=11.
        
        arg_sqrt = norm_space**2 - n_time**2
        
        # Tolerancia muy permisiva para muros tangentes al cono de luz
        if arg_sqrt < -1e-4: 
            # Realmente es time-like (físicamente imposible para un muro de billar válido).
            # Si esto pasa, es que la métrica G está MAL definida (signos o factores).
            return None, None 
        
        if arg_sqrt < 0: arg_sqrt = 0.0

        if abs(n_mink[0]) < 1e-12:
            offset = np.zeros_like(n_mink[1:])
        else:
            denom = norm_space + np.sqrt(arg_sqrt)
            if abs(denom) < 1e-12: return None, None
            offset = (n_mink[0] / denom) * (n_mink[1:] / norm_space)
            
        base_matrix = null_space(n_mink[1:].reshape(1, -1)).T
        return base_matrix, offset
    def get_hopscotch_u(self, u_vec, v_inf, v_0, v_m1):
        center = (v_inf + v_0 + v_m1) / 3.0
        r1 = v_0 - v_inf
        r2 = v_m1 - v_inf
        if np.linalg.norm(r1) < 1e-9: return np.nan
        e1 = r1 / np.linalg.norm(r1)
        proj = np.dot(r2, e1)
        perp = r2 - proj * e1
        norm_perp = np.linalg.norm(perp)
        if norm_perp < 1e-12: return np.nan
        e2 = perp / norm_perp
        
        def to_complex(vec):
            if np.all(vec == 0): return 0j
            x = np.dot(vec, e1)
            y = np.dot(vec, e2)
            z = complex(x, y)
            if abs(z) > 1e-12: z /= abs(z)
            return z

        return self._moebius_cross_ratio(to_complex(u_vec), to_complex(v_inf), to_complex(v_0), to_complex(v_m1))
   
    def get_geodesic_endpoints(self, p, v):
        norm_p = np.linalg.norm(p)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-15: return p, p
        
        cos_theta = abs(np.dot(p, v)) / (norm_p * norm_v) if norm_p > 1e-9 else 1.0
        if abs(cos_theta - 1.0) < 1e-9 or norm_p < 1e-9:
            u = v / norm_v
            return u, -u
            
        e1 = p / norm_p
        v_perp = v - np.dot(v, e1) * e1
        norm_perp = np.linalg.norm(v_perp)
        if norm_perp < 1e-12: return v/norm_v, -v/norm_v
        e2 = v_perp / norm_perp
        
        p_x = norm_p
        v_x, v_y = np.dot(v, e1), norm_perp
        denom = 2 * (p_x * (-v_y)) 
        k = (1 - p_x**2) / denom
        c_2d = np.array([p_x, 0]) + k * np.array([-v_y, v_x])
        r_sq = np.dot(c_2d, c_2d) - 1.0
        if r_sq < 0: r_sq = 0
        r = np.sqrt(r_sq)
        
        d2 = np.dot(c_2d, c_2d)
        d = np.sqrt(d2)
        chord_mid = c_2d / d2 
        h = np.sqrt(max(0, 1.0 - 1.0/d2))
        dir_chord = np.array([-c_2d[1], c_2d[0]]) / d
        u1 = chord_mid + h * dir_chord
        u2 = chord_mid - h * dir_chord
        
        if np.dot(u1 - np.array([p_x, 0]), np.array([v_x, v_y])) > 0:
            u_plus_2d, u_minus_2d = u1, u2
        else:
            u_plus_2d, u_minus_2d = u2, u1
            
        u_plus = u_plus_2d[0] * e1 + u_plus_2d[1] * e2
        u_minus = u_minus_2d[0] * e1 + u_minus_2d[1] * e2
        return u_plus, u_minus
    
    def create_dynamic_slice(self, pos_dewitt, vel_dewitt, view_hint_dewitt=None):
        """
        Crea un slice dinámico robusto.
        - Caso Normal: Plano definido por Posición y Velocidad.
        - Caso Degenerado (V || P): Plano definido por Posición y 'view_hint' (Centro de la cámara).
        """
        p_poinc = self.scale_to_poinc(pos_dewitt)
        v_poinc = self.vect_scale_to_poinc(pos_dewitt, vel_dewitt)
        
        # 1. Verificar colinealidad o velocidad nula
        norm_v = np.linalg.norm(v_poinc)
        use_velocity = True
        
        if norm_v < 1e-9:
            use_velocity = False
        else:
            # Producto cruz en N-dimensiones (o verificar ángulo)
            # Normalizamos temporalmente para chequear paralelismo
            p_u = p_poinc / (np.linalg.norm(p_poinc) + 1e-15)
            v_u = v_poinc / norm_v
            cos_theta = np.clip(np.dot(p_u, v_u), -1.0, 1.0)
            if abs(abs(cos_theta) - 1.0) < 1e-5:
                use_velocity = False

        vectors = [p_poinc]
        
        if use_velocity:
            vectors.append(v_poinc)
        else:
            # --- SOLUCIÓN AL PRIMER ERROR ---
            # Si la velocidad no define un plano, usamos el "Hint" (Centro de la cámara)
            if view_hint_dewitt is not None:
                hint_poinc = self.scale_to_poinc(view_hint_dewitt)
                # Asegurarnos de que el hint no sea también paralelo a P (ej. partícula en el centro)
                vectors.append(hint_poinc)
            else:
                # Fallback extremo: usar un vector arbitrario (eje X)
                fallback = np.zeros_like(p_poinc)
                fallback[0] = 1.0
                vectors.append(fallback)

        # La nueva función complete_orthonormal_basis se encargará de limpiar
        # si el hint también resultara colineal (muy raro).
        basis = self.embedding.complete_orthonormal_basis(vectors, target_dim=2)
        return self.GeodesicHyperplane(self, basis, offset=np.zeros(self.dim))
    def create_vertex_aligned_slice(self, vertex_vector, beta_dim, metric_transform=None, view_hint_dewitt=None, particle_pos_dewitt=None):
        """
        Crea un slice visual que pasa por la Partícula y apunta hacia un Vértice.
        Esto maximiza la "profundidad" de visión dentro de la cámara.
        """
        if metric_transform is not None:
            T = metric_transform
        else:
            T = self.superspace.scale_to_mink_matrix(dim=beta_dim)

        vectors = []
        
        # 1. PUNTAL A: La Partícula (Observador)
        # Si la tenemos, es el ancla principal del slice.
        if particle_pos_dewitt is not None:
             p_poinc = self.scale_to_poinc(particle_pos_dewitt)
             vectors.append(p_poinc)
        else:
             vectors.append(np.zeros(self.dim)) # Fallback al origen

        # 2. PUNTAL B: El Vértice (Objetivo)
        # Proyectamos el vector director del vértice al espacio visual (Poincaré)
        # Usamos scale_to_poinc que maneja correctamente vectores nulos (borde) y temporales (interior).
        v_poinc = self.scale_to_poinc(vertex_vector)
        vectors.append(v_poinc)

        # 3. ORIENTACIÓN: El Centro de la Cámara (Up-Vector)
        # Ayuda a decidir la rotación del plano para que la cámara se vea "ancha".
        if view_hint_dewitt is not None:
             h_poinc = self.scale_to_poinc(view_hint_dewitt)
             vectors.append(h_poinc)
        
        # 4. Generar Base Ortonormal Robusta
        # complete_orthonormal_basis se encargará de rellenar dimensiones si 
        # la partícula y el vértice están alineados (muy raro) o si faltan vectores.
        basis = self.embedding.complete_orthonormal_basis(vectors, target_dim=2)
        
        return self.GeodesicHyperplane(self, basis, offset=np.zeros(self.dim))
    # --- Nested Geodesic Classes ---

    class Geodesic:
        
        def __init__(self, space, p1, p2, n_points=100):
            self.space = space
            self.p1, self.p2 = p1, p2
            self.n_points = n_points
            self.u, self.v, self.center, self.radius = self._compute_geometry()
            self.points = self._generate_points()

        def _compute_geometry(self):
            basis = self.space.embedding.complete_orthonormal_basis([self.p1, self.p2], 2)
            u, v = basis[0], basis[1]
            p1_2d = np.array([np.dot(self.p1, u), np.dot(self.p1, v)])
            p2_2d = np.array([np.dot(self.p2, u), np.dot(self.p2, v)])
            if abs(p1_2d[0]*p2_2d[1] - p1_2d[1]*p2_2d[0]) < 1e-9:
                return u, v, None, None
            M = np.array([p1_2d, p2_2d])
            b = 0.5 * (np.sum(M**2, axis=1) + 1.0)
            try:
                c_2d = np.linalg.solve(M, b)
                c_3d = c_2d[0]*u + c_2d[1]*v
                r = np.sqrt(np.dot(c_3d, c_3d) - 1.0)
                return u, v, c_3d, r
            except np.linalg.LinAlgError:
                return u, v, None, None
        def _generate_points(self):
            if self.radius is None: 
                t = np.linspace(0, 1, self.n_points)
                return self.p1 + np.outer(t, (self.p2 - self.p1))
            else:
                p1_local = self.p1 - self.center
                p2_local = self.p2 - self.center
                phi1 = np.arctan2(np.dot(p1_local, self.v), np.dot(p1_local, self.u))
                phi2 = np.arctan2(np.dot(p2_local, self.v), np.dot(p2_local, self.u))
                if abs(phi2 - phi1) > np.pi:
                    if phi2 > phi1: phi1 += 2*np.pi
                    else: phi2 += 2*np.pi
                t = np.linspace(0, 1, self.n_points)
                phi = phi1 + t * (phi2 - phi1)
                return self.center + self.radius * (np.outer(np.cos(phi), self.u) + np.outer(np.sin(phi), self.v))

    class GeodesicHyperplane:
        
        def __init__(self, space, vectors, offset=None):
            self.space = space
            self.offset = np.array(offset) if offset is not None else np.zeros(space.dim)
            
            # --- FORCE ORTHONORMALIZATION (QR) ---
            # Replaces manual lineal_dep check and manual Gram-Schmidt
            vectors = np.array(vectors)
            if vectors.size > 0:
                Q, _ = np.linalg.qr(vectors.T) # QR on columns
                rank = np.linalg.matrix_rank(vectors)
                self.base = Q[:, :rank].T
            else:
                self.base = np.array(vectors)

            self.sub_dim = len(self.base)
            self.embedding = EuclideanSpace(self.sub_dim)
            if self.space.dim == 3 and self.sub_dim == 2:
                self.normal_vect = np.cross(self.base[0], self.base[1])
            else:
                self.normal_vect = space.embedding.normal_vector(self.base)
            self.center, self.radius = self._radius_and_center()
        def _is_sphere_or_plane(self): return np.all(np.abs(self.offset) < 1e-12)
        def project_to_hyperplane(self, vectors):
            vectors = np.array(vectors)
            dim_space = self.space.dim
            transposed = False
            if vectors.shape[0] == dim_space and vectors.shape[-1] != dim_space:
                vectors = vectors.T
                transposed = True
            elif vectors.ndim == 1 and vectors.shape[0] == dim_space:
                 vectors = vectors[None, :]
            if vectors.ndim == 1: vectors = vectors[None, :] 
            u = np.dot(vectors, self.base[0])
            v = np.dot(vectors, self.base[1])
            coords_2d= np.column_stack([u, v])
            if transposed: return coords_2d.T
            return coords_2d
        def normal_projection(self, vectors):
            coords_2d = self.get_2d_coordinates(vectors)
            proj_3d = np.outer(coords_2d[:, 0], self.base[0]) + np.outer(coords_2d[:, 1], self.base[1])
            return vectors - proj_3d
        def _radius_and_center(self):
            if self._is_sphere_or_plane(): return None, None
            A = np.vstack([self.base, self.offset])
            b_tan = np.dot(self.base, self.offset)
            b_orth = 0.5 * (np.dot(self.offset, self.offset) + 1.0)
            b = np.concatenate([b_tan, [b_orth]])
            try:
                c = np.linalg.lstsq(A, b, rcond=None)[0]
                r = np.sqrt(np.dot(c, c) - 1.0)
                return c, r
            except: return None, None
        
        def get_grid(self, dr=0.25, dphi=np.pi/4, n_points=100):
            if self.sub_dim != 2: return None
            grid_lines = []
            theta = np.linspace(0, 2*np.pi, n_points)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            for r_hyp in np.arange(dr, 5.0, dr):
                rho = np.tanh(r_hyp / 2)
                p_local = np.outer(cos_t, self.base[0]) * rho + np.outer(sin_t, self.base[1]) * rho
                line = [self.space.moebius_add(-self.offset, p) for p in p_local]
                grid_lines.append(np.array(line))
            r_vals = np.linspace(0, 0.9999, n_points)
            for phi in np.arange(0, 2*np.pi, dphi):
                direction = np.cos(phi)*self.base[0] + np.sin(phi)*self.base[1]
                p_local = np.outer(r_vals, direction)
                line = [self.space.moebius_add(-self.offset, p) for p in p_local]
                grid_lines.append(np.array(line))
            return np.array(grid_lines)
        def get_border(self, n_points=100):
            if self.sub_dim != 2: return None
            theta = np.linspace(0, 2*np.pi, n_points)
            # FIX: Unit Circle on Tangent Plane.
            border = np.outer(np.cos(theta),[1,0]) + np.outer(np.sin(theta), [0,1])
            return border
        def get_intesect_points(self, plane, n_points=100):
            if self.sub_dim != 2: return None
            if self._is_sphere_or_plane():
                if plane._is_sphere_or_plane():
                    U, S, Vt = np.linalg.svd(plane.base)
                    n = Vt[-1]
                    r1, r2 = float(np.dot(n, self.base[0])), float(np.dot(n, self.base[1]))
                    if abs(r1) < 1e-12 and abs(r2) < 1e-12: return None
                    u_local = np.array([-r2, r1])
                    x0 = u_local[0]*self.base[0] + u_local[1]*self.base[1]; x0 /= np.linalg.norm(x0)
                    return np.linspace(-1, 1, n_points)[:, None] * x0
                else:
                    offset_nd = plane.offset
                    u, v = self.base[0], self.base[1]
                    coeff_u, coeff_v = np.dot(offset_nd, u), np.dot(offset_nd, v)
                    pc = coeff_u * u + coeff_v * v
                    dir_perp = -coeff_v * u + coeff_u * v
                    norm = np.linalg.norm(dir_perp)
                    if norm < 1e-9: return None
                    t = np.linspace(-0.999, 0.999, n_points)
                    pts = t[:, None] * (dir_perp / norm)
                    return np.array([self.space.moebius_add(-pc, p) for p in pts])
            else:
                if self.center is None or self.radius is None: return None
                c1, r1 = self.center, self.radius
                if plane.center is None:
                    n_wall = plane.normal_vect
                    dist = np.dot(c1, n_wall)
                    if abs(dist) > r1: return None
                    c_int = c1 - dist * n_wall
                    r_int = np.sqrt(r1**2 - dist**2)
                    n_circle = n_wall
                else:
                    c2, r2 = plane.center, plane.radius
                    d_c = np.linalg.norm(c1 - c2)
                    if d_c > r1 + r2 or d_c < abs(r1 - r2) or d_c == 0: return None
                    a = (r1**2 - r2**2 + d_c**2) / (2 * d_c)
                    c_int = c1 + a * (c2 - c1) / d_c
                    r_int = np.sqrt(r1**2 - a**2)
                    n_circle = (c2 - c1) / d_c
                u_arb = np.zeros_like(c_int); u_arb[0] = 1.0
                if abs(np.dot(u_arb, n_circle)) > 0.9: u_arb[1] = 1.0
                v1 = self.space.embedding.graham_schmidt([n_circle, u_arb])[1]
                v1 /= np.linalg.norm(v1)
                base_perp = self.space.embedding.graham_schmidt(np.vstack([n_circle, np.eye(len(c_int))]))
                v2 = base_perp[2] 
                theta = np.linspace(0, 2*np.pi, n_points)
                circ = c_int + r_int * (np.outer(np.cos(theta), v1) + np.outer(np.sin(theta), v2))
                return circ[np.linalg.norm(circ, axis=1) < 1.0 - 1e-5]

class CoxeterGroup:
    def __init__(self, walls, de_witt_space):
        self.walls = walls
        self.space = de_witt_space
        self.n_walls = len(walls)
        
        # Dimensiones
        # Nota: La dimensión del billar hiperbólico es (dim_espacio_config - 1)
        self.billiard_dim = self.space.dim  # d en DeWitt (dimensión de las betas)
        
        # 1. Detectar si es un Simplex
        # Un simplex en H^n tiene n+1 caras. 
        # Aquí self.space.dim es la dimensión de las betas (ej. 3 para 4D).
        # El espacio hiperbólico tiene dim = 2. Un triangulo tiene 3 caras.
        # Por tanto, es simplex si n_walls == billiard_dim.
        self.is_simplex = (self.n_walls == self.billiard_dim)
        print(f"Simplex:{self.is_simplex}")
        self.cartan_matrix = self._calculate_cartan()
        
        # Inicializar variables de finitud
        self.eigenvalues = np.array([])
        self.determinant = 0.0
        self.is_lorentzian = False
        
        if self.is_simplex:
            # --- MÉTODO 1: Criterio Espectral (Vinberg) para Simplices ---
            self.eigenvalues = np.linalg.eigvals(self.cartan_matrix)
            self.determinant = np.linalg.det(self.cartan_matrix)
            # En la convención BKL, firma Lorentziana implica un autovalor negativo
            self.is_lorentzian = (np.sum(self.eigenvalues < -1e-9) == 1)
            
            if self.is_lorentzian:
                self.volume_finity = self._check_finite_volume_simplex()
            else:
                self.volume_finity = False
        else:
            # --- MÉTODO 2: Criterio Geométrico para Poliedros Generales ---
            # No usamos autovalores de Cartan global (matriz sobredimensionada)
            self.volume_finity = self._check_finite_volume_polyhedron()
            self.is_lorentzian = True # Asumimos geometría Lorentziana base

        self.singular_vertices_list = []
        if not self.volume_finity:
            self.singular_vertices_list = self._find_singular_vertices()
    
    def _calculate_cartan(self):
        N = np.array([w.normal for w in self.walls])
        # Producto escalar en métrica DeWitt (Lorentziana)
        Gram = N @ self.space.g_inv @ N.T
        diag = np.diag(Gram)
        with np.errstate(divide='ignore', invalid='ignore'):
            A = 2 * Gram / diag[:, None]
        A[np.isnan(A)] = 0
        return np.rint(A).astype(int)
    
    def _calculate_cartan(self):
        N = np.array([w.normal for w in self.walls])
        Gram = N @ self.space.g_inv @ N.T
        diag = np.diag(Gram)
        with np.errstate(divide='ignore', invalid='ignore'):
            A = 2 * Gram / diag[:, None]
        A[np.isnan(A)] = 0
        return np.rint(A).astype(int)
    def _check_finite_volume_simplex(self):
        """Método clásico de Vinberg: quitar un nodo y chequear si el subgrafo es finito/afín."""
        for k in range(self.n_walls):
            sub_A = np.delete(np.delete(self.cartan_matrix, k, 0), k, 1)
            # Si el subgrafo tiene autovalores negativos, es Indefinido -> No cierra el volumen
            if np.any(np.linalg.eigvals(sub_A) < -1e-7): return False
        return True
    def _check_finite_volume_polyhedron(self):
        """
        Criterio Geométrico General (Corregido):
        El volumen es finito si TODOS los vértices físicos del poliedro están 
        dentro del cono de luz (Time-like) o en el borde (Light-like).
        """
        # Dimensión necesaria para formar un vértice (intersección de d-1 planos)
        required_walls = self.billiard_dim - 1 
        
        if self.n_walls < required_walls: return False

        valid_vertex_found = False
        has_ultra_ideal = False

        # Iterar todas las combinaciones posibles que formen un vértice
        for indices in itertools.combinations(range(self.n_walls), required_walls):
            subset_normals = np.array([self.walls[i].normal for i in indices])
            
            try:
                # SVD para hallar el vector director v (Null space)
                _, _, Vh = np.linalg.svd(subset_normals)
                v = Vh[-1]
            except np.linalg.LinAlgError:
                continue

            # --- CORRECCIÓN CRÍTICA ---
            # Chequeo de orientación (probar v y -v)
            # Usamos producto punto directo (n_i * v^i) SIN MÉTRICA.
            
            other_indices = [i for i in range(self.n_walls) if i not in indices]
            if not other_indices: 
                pass 
            else:
                other_normals = np.array([self.walls[i].normal for i in other_indices])
                
                # ERROR ANTERIOR: projections = other_normals @ self.space.g_inv @ v
                # CORRECCIÓN: Contracción directa Forma-Vector
                projections = other_normals @ v
                
                tol = 1e-7 # Tolerancia un poco más laxa para flotantes
                
                if np.all(projections >= -tol):
                    pass # v es correcto
                elif np.all(projections <= tol):
                    v = -v # -v es correcto
                else:
                    continue # El vértice viola algún muro -> No es parte del poliedro

            # Si llegamos aquí, el vértice es geométricamente válido (pertenece al billar)
            valid_vertex_found = True
            
            # --- CRITERIO DE FINITUD ---
            # Calculamos la norma al cuadrado con la métrica: v . G . v
            # Para esto SÍ usamos la métrica del espacio.
            norm_sq = self.space.squared_norm(v, is_form=False)
            
            # Clasificación:
            # < 0 : Time-like (Interior H^n) -> OK
            # ~ 0 : Light-like (Borde Infinito) -> OK (Cúspide)
            # > 0 : Space-like (Exterior) -> ULTRA-IDEAL -> VOLUMEN INFINITO
            
            if norm_sq > 1e-4: # Tolerancia positiva para descartar ruido numérico en cúspides
                has_ultra_ideal = True
                # Opcional: Break temprano si solo nos importa saber si falla
                # break 
        
        if not valid_vertex_found:
            return False
            
        # Si NO hay vértices ultra-ideales, el volumen está "tapado" y es finito.
        return not has_ultra_ideal
    def find_all_corners(self):
        """
        Encuentra intersecciones de Muros (Corners/Cusps) independientemente de la finitud del volumen.
        Devuelve tanto vértices nulos (infinito) como temporales (finitos).
        """
        verts = []
        beta_dim = self.space.dim + 1 
        
        # Necesitamos al menos D-1 muros para formar un vértice (línea en espacio-tiempo)
        if self.n_walls < beta_dim - 1: return verts

        # Normalización temporal para chequeos de orientación
        unit_normals = []
        for w in self.walls:
            n = w.normal
            norm = np.linalg.norm(n)
            unit_normals.append(n / norm if norm > 1e-12 else n)
        unit_normals = np.array(unit_normals)

        # Iterar combinaciones de muros
        for indices in itertools.combinations(range(self.n_walls), beta_dim - 1):
            subset_normals = np.array([self.walls[i].normal for i in indices])
            
            try:
                # Null space (SVD) nos da el vector director de la intersección
                U, S, Vh = np.linalg.svd(subset_normals)
                v = Vh[-1] 
            except np.linalg.LinAlgError: continue

            # 1. Filtro Físico: El vértice debe estar dentro o en el borde del cono de luz.
            # Norm^2 <= 0 (Time-like) o ~0 (Light-like). 
            # Si es > 0 (Space-like), es un vértice "fuera" del universo físico.
            norm_sq = self.space.squared_norm(v)
            if norm_sq > 1e-3: continue 

            # 2. Filtro de Cámara: El vector debe apuntar hacia DENTRO de la cámara
            # (o al menos no violar ningún muro flagrantemente)
            projs = unit_normals @ v 
            tol = 1e-5
            is_pos = np.all(projs >= -tol)
            is_neg = np.all(-projs >= -tol)
            
            final_v = None
            if is_pos: final_v = v
            elif is_neg: final_v = -v
            else: continue
            
            final_v /= np.linalg.norm(final_v)

            # 3. Deduplicar
            is_dup = False
            for existing in verts:
                if abs(np.dot(existing["vertex_vector"], final_v)) > 1.0 - 1e-3:
                    is_dup = True; break
            
            if not is_dup:
                verts.append({
                    "vertex_vector": final_v, 
                    "description": f"Corner {indices}",
                    "is_infinity": abs(norm_sq) < 1e-3
                })
        
        # Ordenar: Priorizar vértices en el infinito (Cusps) si existen, ya que dan mejor perspectiva
        verts.sort(key=lambda x: x["is_infinity"], reverse=True)
        return verts
    def _find_singular_vertices(self):
        """
        Encuentra direcciones nulas (Cusps) formadas por la intersección de D-1 muros.
        ROBUSTO: Normalización previa para consistencia numérica.
        """
        verts = []
        beta_dim = self.space.dim + 1 
        
        if self.n_walls < beta_dim - 1: return verts

        # Pre-normalizamos todos los vectores normales para el chequeo de ángulos
        # (Solo para este método, no tocamos los muros reales)
        unit_normals = []
        for w in self.walls:
            n = w.normal
            norm = np.linalg.norm(n)
            unit_normals.append(n / norm if norm > 1e-12 else n)
        unit_normals = np.array(unit_normals)

        for indices in itertools.combinations(range(self.n_walls), beta_dim - 1):
            # Usamos las normales originales para el SVD (preserva escala relativa si importa)
            subset_normals = np.array([self.walls[i].normal for i in indices])
            
            try:
                U, S, Vh = np.linalg.svd(subset_normals)
                v = Vh[-1] 
            except np.linalg.LinAlgError: continue

            # 1. Verificación Light-like
            norm_sq = self.space.squared_norm(v, is_form=False)
            if norm_sq > 1e-3: continue # Tolerancia relajada para D altas

            # 2. Orientación con Normales Unitarias
            # Usamos producto Euclídeo simple para chequear "lado del muro"
            # (En el espacio de Betas, la métrica interna define distancias, 
            # pero el signo del producto punto euclídeo define "izquierda/derecha").
            
            projs = unit_normals @ v # Vector de proyecciones
            
            tol = 1e-5
            is_pos = np.all(projs >= -tol)
            is_neg = np.all(-projs >= -tol)
            
            final_v = None
            if is_pos: final_v = v
            elif is_neg: final_v = -v
            else: continue
            
            final_v /= np.linalg.norm(final_v)

            # 3. Chequeo de duplicados
            is_dup = False
            for existing in verts:
                if abs(np.dot(existing["vertex_vector"], final_v)) > 1.0 - 1e-3:
                    is_dup = True; break
            
            if not is_dup:
                verts.append({
                    "vertex_vector": final_v, 
                    "intersecting_walls": indices, 
                    "description": f"Cusp {indices}"
                })
                
        return verts
    def find_fundamental_chamber_center(self):
        normals = np.array([w.normal for w in self.walls])
        if normals.size == 0: return None
        dim = self.space.dim

        # 1. Identificar dirección temporal aproximada (autovector negativo de la métrica)
        vals, vecs = np.linalg.eigh(self.space.g)
        # El autovalor negativo corresponde al tiempo en métrica DeWitt (- + + ...)
        time_idx = np.argmin(vals)
        time_dir = vecs[:, time_idx]
        
        # Asegurar que apunta al futuro (suma positiva es heurística habitual)
        if np.sum(time_dir) < 0: time_dir = -time_dir

        # 2. LP: A_ub * x <= b_ub  =>  -n_i * x <= -1  (o sea n_i*x >= 1)
        A_ub = -normals
        b_ub = -np.ones(len(normals))
        
        bounds = [(-100, 100) for _ in range(dim)]
        
        # TRUCO: Función objetivo = -time_dir
        # Esto fuerza al optimizador a buscar el punto que maximice la componente temporal,
        # empujando la solución "profundo" dentro del cono de luz, lejos de los bordes space-like.
        c = -time_dir 

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                x_sol = res.x
                # Verificar que sea Time-Like (Norma^2 < 0)
                if x_sol @ self.space.g @ x_sol < -1e-4:
                    return x_sol
        except:
            pass
            
        # Fallback: Usar el vector temporal puro (seguro aunque no centrado)
        return time_dir
    def get_data_dict(self):
        return {
            "cartan_matrix": self.cartan_matrix,
            "eigenvalues": self.eigenvalues,
            "determinant": self.determinant,
            "volume_finity": self.volume_finity,
            "singular_vertices": self.singular_vertices_list
        }