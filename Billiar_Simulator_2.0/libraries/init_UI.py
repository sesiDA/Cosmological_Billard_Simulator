import numpy as np
import tkinter as tk
import tkinter.font as tkfont
from .utilities import Form

class init_UI():

    """Initialization window of the simulation program."""

    def __init__(self, title):
        # Sets window and configurates it
        self.w, self.h = 700, 520 # Aumentado ligeramente para el desplegable sin apretar el resto
        
        # Sets internal state variables
        self.closed = False
        self.dilaton_exp_visible = None
        self.coupling_lst_visible = None
        self.pforms_lst_visible = None
        self.form_dilaton_check = None
        self.form_pforms_check = None
        
        # --- LÓGICA DE PARAMETRIZACIÓN INICIAL ---
        self.param_options = ["Raw Kasner initial velocity", "u parametrization"]
        
        vals = [1,0,0]

        self.parameters = {
            "Initial Kasner Exp": vals,
            "Initial Beta Pos": [10, 11, 12],
            "Initial Time": 0.0,
            "Time Speed": 1.0,
            "Dilaton": False,
            "Kasner Dilaton Exp": 0.0,
            "Coupling Constants": [2, 2],
            "P-Forms": False,
            "P-Form List": [1, 2],
            "Diagonal Model": False
        }

        self.type_to_functions = {
            "<class 'int'>":    lambda x: int(x),
            "<class 'float'>":  lambda x: float(eval(str(x))),
            "<class 'list'>":   lambda x: eval(x),
            "<class 'complex'>": lambda x: complex(x),
            "<class 'set'>":    lambda x: set(x),
            "<class 'bool'>":   lambda x: bool(x),
            "<class 'str'>":    lambda x: str(x)
        }

        # Creates window
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)
        self.root.geometry(f"{self.w}x{self.h}") 
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Text Frame
        txt_frame = tk.Frame(self.root, width=self.w, height=self.h/5)
        txt_frame.pack(side="top", fill="x", pady=(10, 0))
        
        title_label = tk.Label(txt_frame, text="Simulation Initialization", font=("Arial", 18))
        title_label.pack(anchor="center")
        
        self.error_msg = tk.Label(txt_frame, text="", font=("Arial", 8), fg="red")
        self.error_msg.pack()

        # --- NUEVO: DROPDOWN DE PARAMETRIZACIÓN ---
        # Lo ponemos justo antes de los formularios para que sea lo primero que se vea
        mode_frame = tk.Frame(self.root)
        mode_frame.pack(side="top", fill="x", padx=30, pady=10)
        
        # Usamos un frame interno centrado
        mode_inner = tk.Frame(mode_frame)
        mode_inner.pack(anchor="center") # Centra el bloque entero
        
        tk.Label(mode_inner, text="Parametrization mode:", font=("Arial", 9, "bold")).pack(side="left")
        self.param_var = tk.StringVar(value=self.param_options[0])
        self.mode_menu = tk.OptionMenu(mode_inner, self.param_var, *self.param_options, command=self.update_kasner_label)
        self.mode_menu.config(width=25, font=("Arial", 8))
        self.mode_menu.pack(side="left", padx=10)

        # Form Frame
        form_frame = tk.Frame(self.root, width=self.w, height=3*self.h/5)
        form_frame.pack(side="top", fill="both", expand=True, padx=20, pady=5)

        form_frame.grid_rowconfigure(0, weight=1) # <--- AÑADE ESTO para expansión vertical
        form_frame.grid_columnconfigure(0, weight=1, uniform="group1")
        form_frame.grid_columnconfigure(1, weight=0) 
        form_frame.grid_columnconfigure(2, weight=1, uniform="group1")
        
        num_frame = tk.Frame(form_frame) # Quita el width fijo, deja que el grid lo maneje
        num_frame.grid(row=0, column=0, sticky="nsew")
        num_frame.grid_columnconfigure(0, weight=1) # <--- AÑADE ESTO

        sep_line = tk.Frame(form_frame, bg="gray", width=1)
        sep_line.grid(row=0, column=1, sticky="ns", pady=10)

        bool_frame = tk.Frame(form_frame) # Quita el width fijo
        bool_frame.grid(row=0, column=2, sticky="nsew")
        bool_frame.grid_columnconfigure(0, weight=1)
        
        # Logic to separate types (Conservative with your original loop)
        self.num_type = []
        self.bool_type = []
        keys_to_exclude = ["Kasner Dilaton Exp", "Coupling Constants", "P-Form List"]
        parameters_type = []

        for key in self.parameters.keys():
            parameters_type.append(type(self.parameters[key]))
            if key == "Diagonal Model": continue
            if key == "Initial Kasner Exp":
                self.parameters[key] = f"{vals}"
            
            if (key in keys_to_exclude) or isinstance(self.parameters[key], bool):
                self.bool_type.append([key, str(self.parameters[key])])
            else:
                self.num_type.append([key, str(self.parameters[key])])
        
        self.parameters_type = parameters_type
        self.forms = {}

        # Creates numeric forms
        for i, (key, value) in enumerate(self.num_type):
            num_frame.grid_rowconfigure(i, weight=1)
            f = Form(num_frame, "·" + key + " =", value)
            f.grid(row=i, column=0, sticky="ew", padx=10)
            self.forms[key] = f

        # Creates boolean forms
        bool_frame.grid_rowconfigure(0, weight=0)
        matter_title = tk.Label(bool_frame, text="Matter Content:", font=("Arial", 12, "bold"))
        matter_title.grid(row=0, column=0, sticky="w", padx=15, pady=(0, 10))
        
        for i, (key, value) in enumerate(self.bool_type):
            row_idx = i + 1
            bool_frame.grid_rowconfigure(row_idx, weight=1)
            if not key in keys_to_exclude:
                f = Form(bool_frame, "-" + key + ":", entry_type="bool")
                f.grid(row=row_idx, column=0, sticky="ew", padx=10)
                f.checkbox.config(command=self.update_dynamic_fields)
            else:
                f = Form(bool_frame, "  ·" + key + ":", value)
            self.forms[key] = f
        
        # --- NUEVO: START Y LOAD BUTTONS ---
        btn_frame = tk.Frame(self.root, width=self.w, height=self.h/5)
        btn_frame.pack(side="bottom", pady=(0, 15))
        
        f_homo = Form(btn_frame, "Diagonal metric", entry_type="bool")
        f_homo.pack(side="top", pady=(5, 5))
        self.forms["Diagonal Model"] = f_homo
        
        # Frame interno para poner botones al lado
        inner_btn_frame = tk.Frame(btn_frame)
        inner_btn_frame.pack(side="top")

        start_button = tk.Button(inner_btn_frame, text="Start", font=("Arial", 14), width=10, command=self.form_is_correct)
        start_button.pack(side="left", padx=5)

        load_button = tk.Button(inner_btn_frame, text="Load Simulation", font=("Arial", 14), width=15, command=self.load_simulation)
        load_button.pack(side="left", padx=5)

        self.root.mainloop()

    # --- NUEVAS FUNCIONES DE LÓGICA ---
    def get_kasner_from_u(self, u_val, dim=3):
        """Implementación robusta de u -> Kasner."""
        if isinstance(u_val, (list, np.ndarray)):
            u_arr = np.array(u_val)
            p = np.zeros(len(u_arr) + 2)
            p[:-2] = u_arr
            rem_sum = 1.0 - np.sum(u_arr)
            rem_sq = 1.0 - np.sum(u_arr**2)
            a, b, c = 2.0, -2.0*rem_sum, rem_sum**2 - rem_sq
            disc = max(0, b**2 - 4*a*c) # max para evitar errores de precisión
            p[-2] = (-b + np.sqrt(disc)) / 2.0
            p[-1] = (-b - np.sqrt(disc)) / 2.0
            return [round(x, 6) for x in p]
        else:
            # Caso 3D estándar (Lifshitz)
            u = float(u_val)
            denom = 1 + u + u**2
            p = [-u/denom, (1+u)/denom, (u*(1+u))/denom]
            return [round(x, 6) for x in p]
    def get_u_from_kasner(self, p_vals):
        """Convierte exponentes p_i a u (escalar o lista)."""
        # Filtramos ceros absolutos para evitar divisiones por cero
        p = np.sort([float(x) for x in p_vals])
        
        if len(p) == 3:
            # p1 <= p2 <= p3. La relación estándar es p1 = -u/R, p2 = (1+u)/R, p3 = u(1+u)/R
            # Por tanto: u = p3 / p2. 
            # Si p2 es 0 (caso [1,0,0]), u tiende a infinito o es un estado de vacío.
            if abs(p[1]) < 1e-7: 
                return 1.0 if abs(p[2]) < 1e-7 else 100.0 # u grande para aproximar p2->0
            u = p[2] / p[1]
            return round(abs(u), 5)
        else:
            # Para N-D devolvemos los componentes independientes
            return [round(x, 5) for x in p[:-2]]
    def update_kasner_label(self, selection):
        """Cambia el texto y realiza la conversión inmediata si es posible."""
        try:
            raw_content = self.forms["Initial Kasner Exp"].get().strip()
            if not raw_content: return
            
            val = eval(raw_content)
            # Detectar dimensión necesaria
            try:
                beta_raw = self.forms["Initial Beta Pos"].get()
                dim = len(eval(beta_raw)) - (1 if self.forms["Dilaton"].get() else 0)
            except: dim = 3

            if selection == "u parametrization":
                self.forms["Initial Kasner Exp"].label.config(text="·Initial u=")
                if isinstance(val, list): # Evita re-convertir si ya es u
                    new_val = self.get_u_from_kasner(val)
                    self.forms["Initial Kasner Exp"].entry.delete(0, tk.END)
                    self.forms["Initial Kasner Exp"].entry.insert(0, str(new_val))
            else:
                self.forms["Initial Kasner Exp"].label.config(text="·Initial Kasner Exp =")
                if not isinstance(val, list) or len(val) < dim:
                    new_val = self.get_kasner_from_u(val, dim)
                    self.forms["Initial Kasner Exp"].entry.delete(0, tk.END)
                    self.forms["Initial Kasner Exp"].entry.insert(0, str(new_val))
        except Exception as e:
            print(f"Aviso: Entrada no convertible momentáneamente ({e})")
    def load_simulation(self):
        """Placeholder para la función de carga."""
        print("Load simulation triggered")
        pass

    def get_kasner_from_u(self, u_input, dim=3):
        """Convierte u (escalar o lista) a exponentes p_i de dimensión dim."""
        try:
            if isinstance(u_input, (float, int)):
                # Caso 3D estándar (Lifshitz)
                u = float(u_input)
                denom = 1 + u + u**2
                return [round(-u/denom, 6), round((1+u)/denom, 6), round((u*(1+u))/denom, 6)]
            else:
                # Caso N-Dimensional
                u_arr = np.array(u_input)
                p = np.zeros(len(u_arr) + 2)
                p[:-2] = u_arr
                rem_sum = 1.0 - np.sum(u_arr)
                rem_sq = 1.0 - np.sum(u_arr**2)
                
                # Ecuación: p1 + p2 = rem_sum | p1^2 + p2^2 = rem_sq
                a, b, c = 2.0, -2.0 * rem_sum, rem_sum**2 - rem_sq
                disc = max(0, b**2 - 4*a*c) # Blindaje contra complejos
                
                p[-2] = (-b + np.sqrt(disc)) / 2.0
                p[-1] = (-b - np.sqrt(disc)) / 2.0
                return [round(x, 6) for x in p]
        except Exception as e:
            print(f"Error interno en get_kasner_from_u: {e}")
            return [1.0] + [0.0]*(dim-1) # Fallback seguro (Kasner puro)
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
        """Asegura que los parámetros salgan de la UI siempre como Kasner (list)."""
        is_u_mode = self.param_var.get() == "u parametrization"
        
        # Primero calculamos la dimensión espacial necesaria para la conversión
        try:
            beta_val = eval(self.forms["Initial Beta Pos"].get())
            # La dimensión de Kasner es la de Beta menos 1 si hay dilatón
            dim_spatial = len(beta_val) - (1 if self.forms["Dilaton"].get() else 0)
        except:
            dim_spatial = 3 # Fallback por seguridad

        for i, key in enumerate(self.parameters.keys()):
            val_raw = self.forms[key].get()
            
            if key == "Initial Kasner Exp" and is_u_mode:
                # TRADUCCIÓN CRÍTICA: Convertimos u a la lista de exponentes p_i
                u_data = eval(val_raw)
                self.parameters[key] = self.get_kasner_from_u(u_data, dim_spatial)
            else:
                # Conversión estándar para el resto de parámetros
                converter = self.type_to_functions[str(self.parameters_type[i])]
                self.parameters[key] = converter(val_raw)

        # Lógica de Dilatón (Post-procesamiento original)
        if self.parameters["Dilaton"]:
            spatial_kasner = self.parameters["Initial Kasner Exp"]
            phi_exp = self.parameters["Kasner Dilaton Exp"]
            # Nos aseguramos de concatenar correctamente para que sea una lista plana
            self.parameters["Initial Kasner Exp"] = list(spatial_kasner) + [phi_exp]
    def _validate_position_against_walls(self, beta, spatial_dim, is_dilaton, p_forms_on):
        """
        Verifica si la posición 'beta' cumple las desigualdades de todos los muros activos.
        Devuelve (True, "") si es válida, o (False, "Mensaje de Error") si falla.
        """
        total_dim = len(beta)
        dil_idx = total_dim - 1 if is_dilaton else -1
        
        # 1. Muros de Simetría (Ordenamiento: b0 < b1 < ... < bn)
        # Esto define la cámara de Weyl gravitatoria básica.
        # Si quieres libertad TOTAL (incluso fuera de la cámara fundamental gravitatoria),
        # puedes comentar este bloque. Pero BKL asume esto habitualmente.
        for i in range(spatial_dim - 1):
            # Condición: beta[i+1] - beta[i] > 0
            if beta[i+1] <= beta[i]:
                return False, f"Symmetry Violation: beta[{i}] must be < beta[{i+1}] (Basic Weyl Chamber)."

        # 2. Muro Gravitatorio Dominante (2*b0 + sum(others) > 0)
        # Solo verificamos el muro más restrictivo cerca de la singularidad
        val_grav = 2.0 * beta[0]
        for k in range(1, spatial_dim):
            val_grav += beta[k]
        
        if val_grav <= 0:
            return False, "Gravity Wall Violation: Point is not in the physical Time-like region."

        # 3. Muros de P-Formas (La parte crítica)
        if p_forms_on:
            try:
                # Recuperamos los datos crudos del formulario
                raw_p_list = self.forms["P-Form List"].get()
                raw_couplings = self.forms["Coupling Constants"].get()
                
                p_list = list(eval(raw_p_list))
                couplings = eval(raw_couplings)
                if isinstance(couplings, (int, float)): couplings = [couplings]
                
                for i, p in enumerate(p_list):
                    coupling = couplings[i] if i < len(couplings) else couplings[-1]
                    
                    # --- Muro Eléctrico ---
                    # Desigualdad: Sum(beta_0...beta_p-1) - 0.5 * C * phi > 0
                    if p <= spatial_dim:
                        wall_val = sum(beta[k] for k in range(p))
                        if is_dilaton:
                            wall_val -= 0.5 * coupling * beta[dil_idx]
                        
                        if wall_val <= 0:
                            return False, f"Electric {p}-Form Wall Violation. (Try increasing betas or adjusting dilaton)."

                    # --- Muro Magnético ---
                    # Desigualdad: Sum(beta_0...beta_q-1) + 0.5 * C * phi > 0
                    p_magn = spatial_dim - p - 2
                    if p_magn > 0:
                        wall_val = sum(beta[k] for k in range(p_magn))
                        if is_dilaton:
                            wall_val += 0.5 * coupling * beta[dil_idx]
                            
                        if wall_val <= 0:
                            return False, f"Magnetic {p}-Form (dual {p_magn}) Wall Violation."
                            
            except Exception as e:
                return False, f"Error validating P-Forms: {e}"

        return True, ""
    def form_is_correct(self):
    
        """Checks if init form is correct."""
    
        incorrect_message=""
        is_u_mode = self.param_var.get() == "u parametrization"
        for i, expected_type in enumerate(self.parameters_type):
            key_name = list(self.parameters.keys())[i]
            param_value = list(self.forms.values())[i].get()
            
            #Looks at the format 
            try:
                if key_name == "Initial Kasner Exp" and is_u_mode:
                    u_val = eval(param_value)
                    converted_val = self.get_kasner_from_u(u_val)
                else:
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
                    
                    # --- Preparación de Datos ---
                    try:
                        kasner_raw = self.forms["Initial Kasner Exp"].get()
                        is_u_mode = self.param_var.get() == "u parametrization"
                        if is_u_mode:
                            u_eval = eval(kasner_raw)
                            kasner_dim = 3 if isinstance(u_eval, (int, float)) else len(u_eval) + 2
                        else:
                            kasner_dim = len(list(eval(kasner_raw)))
                        is_dilaton = self.forms["Dilaton"].get()
                        is_pforms = self.forms["P-Forms"].get()
                        
                        expected_dim = kasner_dim + (1 if is_dilaton else 0)
                    except:
                        incorrect_message = "Invalid Kasner Exponents format."
                        break

                    # 1. Chequeo de Dimensión
                    if len(beta) != expected_dim:
                        incorrect_message = f"Beta Pos dimension mismatch. Expected {expected_dim} (Space + Dilaton)."
                    
                    # 2. Chequeo Timelike (Métrica de DeWitt)
                    else:
                        sq_norm = np.sum(beta**2)
                        if is_dilaton:
                            sq_norm_grav = np.sum(beta[:-1]**2)
                            linear_grav = np.sum(beta[:-1])**2
                            # Check gravitatorio parcial
                            if (sq_norm_grav - linear_grav) > 0:
                                # Esto es solo una advertencia, no bloqueante si no quieres
                                pass 
                        else:
                            if (sq_norm - np.sum(beta)**2) >= 0:
                                incorrect_message = "Position must be Timelike (inside Light Cone)."

                    # 3. VALIDACIÓN DINÁMICA DE MUROS (Aquí ocurre la magia)
                    if incorrect_message == "":
                        is_valid, msg = self._validate_position_against_walls(beta, kasner_dim, is_dilaton, is_pforms)
                        if not is_valid:
                            incorrect_message = msg
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
                        if self.forms["Dilaton"].get() and len(converted_val) != len(couplings):
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