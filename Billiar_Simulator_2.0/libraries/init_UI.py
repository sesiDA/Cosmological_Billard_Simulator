import numpy as np
import tkinter as tk
import tkinter.font as tkfont
from .utilities import Form

class init_UI():

    """Initialization window of the simulation program."""

    def __init__(self,title):
        #Sets window and configurates it
        self.w,self.h=700,500
        
        #Sets internal state variables
        self.closed=False
        self.dilaton_exp_visible = None
        self.coupling_lst_visible = None
        self.pforms_lst_visible = None
        self.form_dilaton_check = None
        self.form_pforms_check = None
        
        u = 1/np.pi # approx 0.3183
        p1=1/np.pi
        p2=1/np.e
        def get_kasner_param_3d(u):
            
            denom = 1 + u + u**2
            p1 = -u / denom
            p2 = (1 + u) / denom
            p3 = (u * (1 + u)) / denom
            return [p1,p2,p3]
            
        def get_kasner_param_4d(p1,p2):
            sum_remaining = 1.0 - (p1 + p2)
            sq_sum_remaining = 1.0 - (p1**2 + p2**2)
            a = 2.0
            b = -2.0 * sum_remaining
            c = sum_remaining**2 - sq_sum_remaining
            discriminante = b**2 - 4*a*c
            sqrt_disc = np.sqrt(discriminante)
            p3 = (-b + sqrt_disc) / (2*a)
            p4 = (-b - sqrt_disc) / (2*a)
            return [p1,p2,p3,p4]
        vect10D=[1,0,0,0,0,0,0,0,0,0,0]
        # Ordenamos para mantener consistencia con la cámara de Weyl estándar (p1 < p2 < p3)
        # Aunque la fórmula de u ya suele dar p1 negativo.
        vals =sorted(get_kasner_param_4d(p1,p2))
        #vals=vect10D
        #vals=[0,1]
        self.parameters={"Initial Kasner Exp":vals,
                         "Initial Beta Pos":[10,11,12,13,14,15,16,17,18],
                         "Initial Time":0.0,
                         "Time Speed":1.0,
                         "Dilaton":False,
                         "Kasner Dilaton Exp":0.0,
                         "Coupling Constants":[2,2],
                         "P-Forms":False,
                         "P-Form List":[1,2],
                         "Homogeneous Model": False}
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
            if key == "Homogeneous Model": 
                continue
            if key=="Initial Kasner Exp":
                self.parameters[key]=f"{vals}"
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
        f_homo = Form(btn_frame, "Homogeneous Model", entry_type="bool")
        f_homo.pack(side="top", pady=(5, 5))
        self.forms["Homogeneous Model"] = f_homo
        
        start_button = tk.Button(btn_frame, text="Start", font=("Arial", 14),command=self.form_is_correct)
        start_button.pack(side="top", pady=(0, 10))

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
                    
                    # --- Preparación de Datos ---
                    try:
                        kasner_raw = self.forms["Initial Kasner Exp"].get()
                        kasner_dim = len(list(eval(kasner_raw))) # Dimensión espacial
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
                    # Sum(b^2) - (Sum(b))^2 < 0. Esto sigue siendo necesario para que exista proyección hiperbólica.
                    # IMPORTANTE: Calcula esto con la métrica SIN normalización extra primero, o usa la fórmula genérica.
                    # La fórmula genérica para G_ij = delta - 1 es: sum(x^2) - sum(x)^2.
                    # Si tienes dilaton, la métrica es diferente. 
                    # Para simplificar, usamos la condición Lorentziana básica del espacio de configuración.
                    else:
                        # Calculamos métrica aproximada para validar Lorentziano
                        sq_norm = np.sum(beta**2)
                        if is_dilaton:
                            # Asumimos que el input está normalizado para G_phi=1 o 2
                            # Si no estamos seguros, relajamos este check o usamos una cota laxa
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