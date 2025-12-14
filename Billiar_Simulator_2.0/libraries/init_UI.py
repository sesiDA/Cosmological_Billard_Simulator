import numpy as np
import tkinter as tk
import tkinter.font as tkfont
from .utilities import Form

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
        
        p1=-1/np.pi
        p2=1/np.e
        
        S=1-p1-p2
        Q=1-p1**2-p2**2
        D=np.sqrt(2*Q-S**2)
        p3=(S+D)/2
        p4=(S-D)/2
    
        # Ordenamos para mantener consistencia con la cámara de Weyl estándar (p1 < p2 < p3)
        # Aunque la fórmula de u ya suele dar p1 negativo.
        vals =sorted([p1, p2, p3,p4])
        self.parameters={"Initial Kasner Exp":vals,
                         "Initial Beta Pos":[1,3,4,5],
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