import textwrap
import numpy as np
import tkinter as tk
import tkinter.font as tkfont
import matplotlib.pyplot as plt
from tkinter import ttk
from .utilities import Form,GraphicDisplay
from .physics_libs import SimulationCore

class SimInfo():
    
    """This pannel shows in real time all important and some optional information of the simulation."""

    def __init__(self, root, init_parameters, width=0, height=0):
        self.frame = tk.Frame(root, width=width, height=height, bd=1, relief="solid")
        self.fmt = self._format_value 
        if width > 0 or height > 0:
            self.frame.pack_propagate(False)
        
        self.label_font = tkfont.Font(family="Arial", size=9, weight="bold")
        self.value_font = tkfont.Font(family="Arial", size=9)
        self.title_font = tkfont.Font(family="Arial", size=14, weight="bold")
        self.courier_font = tkfont.Font(family="Courier New", size=8)    
        
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
        
        #Row creator(helper)
        def create_row(row_idx, text):
            tk.Label(self.data_frame, text=text, font=self.label_font).grid(row=row_idx, column=0, sticky="w", pady=2)
            lbl = tk.Label(self.data_frame, text="---", font=self.value_font)
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

        #Cartan matrix
        tk.Label(self.data_frame, text="Cartan Matrix:", font=self.label_font).grid(row=8, column=0, sticky="nw", pady=2)
        self.lbl_cartan = tk.Label(self.data_frame, text="Computing...", font=self.courier_font, justify="left") 
        self.lbl_cartan.grid(row=8, column=1, sticky="w", padx=5)

        #Cartan Matrix
        self.lbl_chaos_status= create_row(7,"Volume Finity:")
        
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
    def _set_text(self, label_widget, text):
        
        """Sets the text in the label_widget in the correct size to ocupy the correct amount of screen based on the text. If too much text sets it in the next line"""
        text = str(text).replace('\x00', '')
        label_widget.config(text=text)  
    def _format_value(self, value):
        
        """Puts in the right format numbers or arrays.Uses cientific notation (.3e) just if >= 1000 or < 0.001. Elsewhere uses normal format(.4g)."""
        
        def fmt_item(x):
            try:
                fx = float(x)
                if fx == 0: return "0.0"
                if abs(fx) >= 1000 or abs(fx) < 0.001: return f"{fx:.2g}"
                else: return f"{fx:.4g}"
            except: return str(x)
        try:
            if hasattr(value, '__iter__') and not isinstance(value, str):
                return "(" + ", ".join(fmt_item(x) for x in value) + ")"
            else: return fmt_item(value)
        except: return str(value)
    def update_info(self, tau=None, t_val=None, vel_kasner=None, vel_poinc=None, pos_beta=None, pos_poinc=None, chaos_data=None):
        if tau is not None:
            self._set_text(self.lbl_tau, self._format_value(tau))
            
            # CAMBIO: Usar t_val si viene, sino calcular (fallback)
            if t_val is not None:
                self._set_text(self.lbl_t, self._format_value(t_val))
            else:
                try: self._set_text(self.lbl_t, self._format_value(np.exp(-float(tau))))
                except: self.lbl_t.config(text="Inf")
                
        if vel_kasner is not None: self._set_text(self.lbl_vel_kasner, self._format_value(vel_kasner))
        if vel_poinc is not None: self._set_text(self.lbl_vel_poinc,  self._format_value(vel_poinc))
        if pos_beta is not None: self._set_text(self.lbl_pos_beta,    self._format_value(pos_beta))
        if pos_poinc is not None: self._set_text(self.lbl_pos_poinc,  self._format_value(pos_poinc))

        if chaos_data:
            # Solo actualizamos si realmente cambia algo importante, pero para simplificar,
            # lo dejamos directo ya que _set_text es barato ahora.
            matrix = chaos_data.get("cartan_matrix", None)
            if matrix is not None:
                 mat_str = np.array2string(matrix, separator=', ', max_line_width=100).replace('[', '').replace(']', '')
                 self.lbl_cartan.config(text=mat_str)

            status = chaos_data.get("volume_finity", "Unknown")
            self._set_text(self.lbl_chaos_status, status)
            self.lbl_chaos_status.config(fg="green" if status else "red")
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
        
        if self.sim.beta_dim<4:
            init_hopscotch = [0.0, 0.0] 
            gd_hopscotch = GraphicDisplay(
                self.scrollable_frame,
                data=init_hopscotch,
                init_time=self.sim.tau,
                update=self.sim.get_hopscotch_frame_data, # <--- TU NUEVA FUNCIÓN
                plot_type="hopscotch",                    # <--- EL NUEVO TIPO
                title="Hopscotch Diagram (Kasner Epochs)",
                graph_title="u- vs u+",
                axis_titles=["u- (Past)", "u+ (Future)"],
                width=gd_width,
                height=gd_height,
                grid=True,
                time_vel=self.time_vel,
                on_step_callback=None,
                on_close_callback=self.remove_graphic,
                on_amplify_callback=self.toggle_amplify_graphic
            )
            self.graphic_displays.append(gd_hopscotch)

            total_plots = len(self.graphic_displays) - 1 
            row_h = total_plots // n_per_row
            col_h = total_plots % n_per_row
            gd_hopscotch.grid(row=row_h, column=col_h, padx=10, pady=10, sticky="nsew")
            
            
        row_ts = self.sim.dim // n_per_row + 1 
        col_ts = 0
        logs=self.sim.get_history_data()
        gd_history = GraphicDisplay(
            self.scrollable_frame,
            data=(logs[0], logs[1]),
            init_time=self.sim.tau,
            update=self.sim.get_history_data, 
            plot_type="multiplot",        
            title="Evolution of Kasner axes",
            axis_titles=["Coordinate Time (t)", "Scale Factors"],
            width=gd_width,
            height=gd_height,
            grid=True, 
            on_step_callback=None,
            on_close_callback=self.remove_graphic,
            on_amplify_callback=self.toggle_amplify_graphic)
        self.graphic_displays.append(gd_history)
        gd_history.grid(row=row_ts, column=col_ts, padx=10, pady=10, sticky="nsew")

        for i in range(len(self.sim.slices)):
            plot_title = "Dynamic Slice" if i == 0 else f"Inspection Slice {i}"
            row = i // n_per_row
            col = i % n_per_row
            
            particle_pos, walls, border, grid = self.sim.get_part_pos(i), self.sim.get_walls(i), self.sim.get_border(i), self.sim.get_grid(i)
            slice_def = self.sim.get_slice_def(i)
            
            full_slice_data = self.sim._get_slice_data(i)
            initial_prob_data = full_slice_data.get("prob_dist", None)
            
            def fmt_vec(v): 
                return f"[{', '.join(f'{x:.2f}' for x in v)}]"
            base_u = fmt_vec(slice_def[0][0])
            base_v = fmt_vec(slice_def[0][1])
            offset_str = fmt_vec(slice_def[1])

            is_dynamic_slice = (i == 0 and self.sim.beta_dim > 3)
            use_trail = not is_dynamic_slice
            
            gd = GraphicDisplay(
                self.scrollable_frame,
                particle_pos,
                init_time=self.sim.tau,
                wall_data=walls,
                update=self.sim.update if i==0 else self.sim.fast_update,
                plot_type="scatter",
                color="red",
                time_vel=self.time_vel,
                title=plot_title,
                graph_title=f"Offset={offset_str}",
                axis_titles=[f"Dir {base_u}", f"Dir {base_v}"],
                width=gd_width, 
                height=gd_height,
                grid=grid,
                border=[border[:, 0], border[:, 1]],
                fargs=i,
                on_step_callback=self.relay_to_ui if i==0 else None,
                on_close_callback=self.remove_graphic,
                on_amplify_callback=self.toggle_amplify_graphic,
                enable_trail=use_trail,
                prob_data=initial_prob_data
            )
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
        
        # 1. Comprobar si este gráfico era el portador de actualizaciones
        was_updater = (target_graphic.on_step_callback is not None)
        
        # Erase graphic from tkinter memory
        target_graphic.frame.destroy()
        
        # Erase graphic from list of graphics
        if target_graphic in self.graphic_displays:
            self.graphic_displays.remove(target_graphic)
        
        # If one graphic was amplified reestarts last postion
        if self.amplified_graphic == target_graphic:
            self.amplified_graphic = None

        # 2. Si era el portador, asignar el rol a otro
        if was_updater and self.graphic_displays:
            self._assign_new_updater()

        # Redraws grid
        self.redraw_grid()

    def _assign_new_updater(self):
        """Asigna el rol de actualizar la UI a un nuevo gráfico superviviente."""
        if not self.graphic_displays: return
        
        # Prioridad 1: Buscar un gráfico de tipo 'scatter' (Billar)
        candidate = None
        for gd in self.graphic_displays:
            if gd.plot_type == "scatter":
                candidate = gd
                break
        
        # Asignar callback y PODER DE SIMULACIÓN
        if candidate:
            print(f"Transferring UI update role to: {candidate.lbl_title.cget('text')}")
            
            # 1. Asignar el callback de UI (para pintar los paneles laterales)
            candidate.on_step_callback = self.relay_to_ui
            
            # 2. CRÍTICO: Asignar la función que avanza la física
            # Si el candidato no tenía función de update (era pasivo), le damos la del sim.
            # Esto es vital porque solo el gráfico "activo" empuja el tiempo tau.
            candidate.update_func = self.sim.update
            
            # 3. Asegurarse de que el nuevo portador tenga la velocidad correcta
            candidate.time_vel = self.time_vel
            
            # 4. Asegurarse de que no esté pausado accidentalmente si la simulación debe correr
            if not self.sim_mattres.stop if hasattr(self, 'sim_mattres') else True: 
                 # Nota: Aquí accedemos al estado global. Como 'toggle_playback' afecta a todos,
                 # asumimos que si uno corre, todos corren.
                 candidate.play()
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
    def relay_to_ui(self, current_time):
        """Function that returns all information data and curring time to all other ui pannels when Callback"""
        if self.callback_update_ui:
            coxeter_data = self.sim.coxeter_group_params
            p = self.sim.particle

            current_t = self.sim.t 
            
            self.callback_update_ui(
                tau=current_time,
                t_val=current_t,
                vel_kasner=p.kasner_vel,
                vel_poinc=self.sim.space.vect_scale_to_poinc(p.pos, p.kasner_vel),
                pos_beta=p.pos,
                pos_poinc=self.sim.space.scale_to_poinc(p.pos),
                coxeter_data=coxeter_data
            )
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
        self.sim.set_time(new_time)
        for display in self.graphic_displays:
            display.set_time(new_time)
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
        initial_tau = self.sim_mattres.sim.tau
        self.sim_mattres.relay_to_ui(initial_tau)
        self.root.mainloop()
    def _distribute_updates(self, tau, t_val, vel_kasner, vel_poinc, pos_beta, pos_poinc, coxeter_data):
        """Updates all simulation information in all pannels"""
        self.info_panel.update_info(
            tau=tau, 
            t_val=t_val, # <--- PASARLO AQUÍ
            vel_kasner=vel_kasner, 
            vel_poinc=vel_poinc,
            pos_beta=pos_beta, 
            pos_poinc=pos_poinc,
            chaos_data=coxeter_data
        )
        self.control_panel.update_time_displays(tau)
    def on_close(self):
        """Manages closing all elements wen window is closed"""
        plt.close('all')
        self.root.destroy()