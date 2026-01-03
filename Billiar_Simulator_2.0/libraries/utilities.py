import numpy as np
import tkinter as tk
import tkinter.font as tkfont
import textwrap
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from collections import deque

class Graphic:
    """
    Motor gráfico optimizado. 
    Soluciona el problema de la pausa mediante 'force_update' y 
    gestiona la actualización dinámica de geometría (muros) y datos (partículas).
    """

    def __init__(self, root, data, init_time=0.0, wall_data=None, update=None,
                 lims=[None, None], title="", axis_titles=["", ""], grid=False,
                 plot_type="plot", color="black", point_size=20, time_vel=1,
                 border=False, fargs=None, on_step_callback=None, enable_trail=True,prob_data=None):
        
        # --- Configuración ---
        self.plot_type = plot_type
        self.update_func = update
        self.fargs = (fargs,) if fargs is not None else ()
        self.on_step_callback = on_step_callback
        self.enable_trail = enable_trail
        
        # Estado
        self.is_running = False
        # CORRECCIÓN 1: Usar Diccionario para evitar duplicados en la leyenda
        self.legend_handles = {} 
        self.is_amplified = False
        self.current_frame = init_time
        self.time_vel = float(time_vel)
        self.step_accumulator = 0.0
        self.dt_simulation = 0.05
        
        # --- Matplotlib Setup ---
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        
        self.ax.tick_params(axis='both', which='major', labelsize=8)
        if title: self.ax.set_title(title, fontsize=10)
        if axis_titles[0]: self.ax.set_xlabel(axis_titles[0], fontsize=8)
        if axis_titles[1]: self.ax.set_ylabel(axis_titles[1], fontsize=8)

        self._setup_grid_and_border(grid, border)
        self._setup_limits(lims, plot_type)

        self.artists = {
            "particle": None,
            "walls": [],
            "lines": [],
            "trail": None,
            "heatmap": None  
        }
        
        self.trail_data_x = [] 
        self.trail_data_y = [] 
        
        # Inicialización según tipo
        if plot_type == "scatter" or plot_type == "plot":
            self._setup_billiard(data, wall_data, color, point_size, prob_data)
        elif plot_type == "multiplot":
            self._setup_multiplot(data)
        elif plot_type == "hopscotch": 
            self._setup_hopscotch(data, color, point_size)

        self.animation = FuncAnimation(
            self.fig, 
            self._internal_update, 
            frames=self._frame_generator, 
            interval=50, 
            blit=False, 
            cache_frame_data=False
        )
        self.legend_created = False
    
    def pack(self):
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    
    def _setup_grid_and_border(self, grid, border):
        if isinstance(grid, np.ndarray):
            for line in grid:
                self.ax.plot(line.T[0], line.T[1], color="lightgray", linewidth=0.5, zorder=0)
        elif isinstance(grid, bool) and grid:
            self.ax.grid(True, linestyle='--', alpha=0.6)
        if isinstance(border, (list, np.ndarray, tuple)):
            self.ax.plot(border[0], border[1], color="black", linewidth=1.5, zorder=1)
    def _setup_limits(self, lims, plot_type):
        if plot_type == "multiplot":
            self.ax.set_xscale("log"); self.ax.set_yscale("log")
            self.ax.set_xlim(1, 1e-12); self.ax.set_ylim(1e-1, 1)
        elif plot_type == "hopscotch":
            self.ax.set_xlim(-2.5, 2.5); self.ax.set_ylim(-2.5, 2.5)
            self.ax.set_aspect('equal', adjustable='datalim')
        else:
            if lims[0]: self.ax.set_xlim(lims[0])
            if lims[1]: self.ax.set_ylim(lims[1])
            self.ax.set_aspect('equal', adjustable='datalim')
    def _setup_billiard(self, data, wall_data, color, point_size, prob_data=None):
        # --- Lógica de Partícula (Existente) ---
        if not data is None:
            x, y = data[0], data[1]
            self.artists["particle"] = self.ax.scatter([x], [y], color=color, s=point_size, zorder=10)
        
        # --- Lógica de Muros (Existente) ---
        if wall_data:
            self._update_walls_and_legend(wall_data)
        
        self.artists["trail"], = self.ax.plot([], [], color='red', alpha=0.3, linewidth=1)

        # --- LÓGICA DE HEATMAP (NUEVA) ---
        if prob_data is not None:
            U = prob_data["U"]
            V = prob_data["V"]
            Z = prob_data["Z"]
            
            self.artists["heatmap"] = self.ax.pcolormesh(
                U, V, Z,  
                cmap='inferno', 
                zorder=0, 
                alpha=0.7,
                vmin=np.nanmin(Z), 
                vmax=np.nanmax(Z)
            )
    def _setup_multiplot(self, data):
        if data is None or len(data) < 2 or len(data[0]) == 0: return
        times, positions = data[0], data[1]
        colors = plt.cm.jet(np.linspace(0, 1, positions.shape[1]))
        for i in range(positions.shape[1]):
            # Etiqueta por defecto: a, b, c... o Axis 0, Axis 1...
            # Usamos letras minúsculas para seguir la convención de Kasner (a,b,c)
            label_text = f"Scale Factor: Axis {i}"
            
            # Crear la línea
            line, = self.ax.plot(times, positions[:, i], color=colors[i])
            self.artists["lines"].append(line)
            
            # --- AÑADIR A LEYENDA (NUEVO) ---
            if label_text not in self.legend_handles:
                # Creamos un proxy artist para la leyenda
                h = mlines.Line2D([], [], color=colors[i], linewidth=2, label=label_text)
                self.legend_handles[label_text] = h
    def _setup_hopscotch(self, data, color, point_size):
        INF_VIS = 100 
        regions_def = [
            ((-1, 0),  1,  INF_VIS, "magenta",  "B_ba"),
            ((-1, -INF_VIS), 1, INF_VIS-1, "skyblue",   "B_bc"),
            ((0, -1),  INF_VIS, 1,  "brown",   "B_ab"),
            ((0, -INF_VIS), INF_VIS, INF_VIS-1, "green","B_ac"),
            ((-INF_VIS, 0), INF_VIS-1, INF_VIS,"yellow", "B_ca" ),
            ((-INF_VIS, -1), INF_VIS-1, 1, "pink",   "B_cb"),
        ]
        for (xy, w, h, c, lbl) in regions_def:
            rect = patches.Rectangle(xy, w, h, linewidth=0, facecolor=c, alpha=0.15)
            self.ax.add_patch(rect)
            self.ax.text(max(min(xy[0] + w/2, 2), -2), max(min(xy[1] + h/2, 2), -2), lbl, 
                         fontsize=8, ha='center', va='center', alpha=0.6, color="black" ,fontweight='bold')
        self.ax.axvline(x=0, color='gray', linewidth=0.5)
        self.ax.axvline(x=-1, color='gray', linewidth=0.5)
        self.ax.axhline(y=0, color='gray', linewidth=0.5)
        self.ax.axhline(y=-1, color='gray',linewidth=0.5)
        
        ux, uy = (0,0)
        if data is not None and len(data) >= 2: ux, uy = data[0], data[1]
        self.artists["particle"] = self.ax.scatter([ux], [uy], color='red', s=point_size*2, zorder=20, edgecolors="black")
        self.hopscotch_info_text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        self.artists["trail"], = self.ax.plot([], [], color='red', alpha=0.3, linewidth=1)

    def _frame_generator(self):
        while True:
            yield self.current_frame
            if self.is_running: self.current_frame += self.dt_simulation * self.time_vel
    def _internal_update(self, frame, force_update=False):
        if not self.is_running and not force_update: return []
        raw_data = None
        steps_to_run = 0
        if force_update:
            steps_to_run = 1
            target_tau_for_step = self.current_frame 
        else:
            self.step_accumulator += self.time_vel
            steps_to_run = int(self.step_accumulator)
            MAX_STEPS_PER_FRAME = 20
            if steps_to_run > MAX_STEPS_PER_FRAME:
                steps_to_run = MAX_STEPS_PER_FRAME
                if self.step_accumulator > MAX_STEPS_PER_FRAME * 2: self.step_accumulator = MAX_STEPS_PER_FRAME * 2
            if steps_to_run == 0: return []
            self.step_accumulator -= steps_to_run
            target_tau_for_step = self.current_frame
        
        for _ in range(steps_to_run):
            if not force_update: target_tau_for_step += self.dt_simulation
            raw_data = self.update_func(target_tau_for_step, *self.fargs) if callable(self.update_func) else None
            if not force_update: self.current_frame = target_tau_for_step
            if raw_data is None: break
            
        if self.on_step_callback and (self.is_running or force_update): self.on_step_callback(self.current_frame)
        if raw_data is None: return []
        
        if self.plot_type == "scatter" or self.plot_type == "plot": self._update_billiard(raw_data)
        elif self.plot_type == "multiplot": self._update_multiplot(raw_data)
        elif self.plot_type == "hopscotch": self._update_hopscotch(raw_data)    
        return []

    # --- MÉTODO AUXILIAR PARA MUROS Y LEYENDA (Fix para problema 1 y 2) ---
    def _update_walls_and_legend(self, walls_data):
        lines = self.artists["walls"]
        new_labels_found = False

        for i, wall_info in enumerate(walls_data):
            if isinstance(wall_info, dict):
                pts = wall_info['data']
                c = wall_info.get('color', 'black')
                lbl = wall_info.get('label', None)
                
                # CORRECCIÓN 1: Usar Diccionario para evitar duplicados
                if lbl and lbl not in self.legend_handles:
                    h = mlines.Line2D([], [], color=c, linewidth=2, label=lbl)
                    self.legend_handles[lbl] = h
                    new_labels_found = True
            else: 
                pts = wall_info; c = 'black'
            
            # CORRECCIÓN 2: Asegurar visibilidad al reutilizar líneas
            if i < len(lines):
                lines[i].set_data(pts[:, 0], pts[:, 1])
                lines[i].set_color(c)
                lines[i].set_visible(True) # ¡Importante!
            else:
                l, = self.ax.plot(pts[:, 0], pts[:, 1], color=c, lw=2.0, zorder=5)
                lines.append(l)
        
        # Ocultar sobrantes
        for k in range(len(walls_data), len(lines)): 
            lines[k].set_visible(False)
            
        return new_labels_found
    def _update_billiard(self, raw_data):
        if not isinstance(raw_data, dict): return
        particle_pos = raw_data.get("particle")
        walls_data = raw_data.get("walls")
        slice_def = raw_data.get("slice_def")
        
        # --- LÓGICA DEL HEATMAP MODIFICADA ---
        prob_data = raw_data.get("prob_dist", None)
        
        if prob_data is not None:
            # CASO 1: Recibimos datos -> Creamos o Mostramos
            U = prob_data["U"]
            V = prob_data["V"]
            Z = prob_data["Z"]
            
            if self.artists["heatmap"] is None:
                self.artists["heatmap"] = self.ax.pcolormesh(
                    U, V, Z, 
                    cmap='inferno', 
                    zorder=0, 
                    alpha=0.7,
                    vmin=np.nanmin(Z), 
                    vmax=np.nanmax(Z)
                )
            else:
                # Si existe, aseguramos que sea visible
                self.artists["heatmap"].set_visible(True)
                # Si el grid cambiase (cosa que en inspection slice no pasa), aquí haríamos set_array
        else:
            # CASO 2: NO recibimos datos -> Ocultamos
            if self.artists["heatmap"] is not None:
                self.artists["heatmap"].set_visible(False)

        # Actualización de partícula
        if particle_pos is not None and self.artists["particle"]:
            self.artists["particle"].set_offsets(np.column_stack(particle_pos))
        
        # Actualización de estela
        if self.enable_trail:
            self.trail_data_x.append(particle_pos[0])
            self.trail_data_y.append(particle_pos[1])
            self.artists["trail"].set_data(self.trail_data_x, self.trail_data_y)
        
        # Actualización de Muros y Leyenda
        new_labels = False
        if walls_data is not None:
            new_labels = self._update_walls_and_legend(walls_data)
        
        if self.is_amplified and new_labels:
            pass

        if slice_def is not None:
            base, offset = slice_def
            str_offset = self._format_vector(offset)
            self.ax.set_title(f"Offset={str_offset}", fontsize=9)
    def _update_multiplot(self, data):
        if data is None or len(data) < 2 or len(data[0]) == 0: return
        times = np.array(data[0])
        positions = np.array(data[1])
        if len(times) == 0: return

        for i, line in enumerate(self.artists["lines"]):
            if i < positions.shape[1]:
                line.set_data(times, positions[:, i])

        t_min, t_max = times[0], times[-1]
        FLOAT_LIMIT = 1e-300 
        if t_max <= FLOAT_LIMIT: t_max = FLOAT_LIMIT
        if t_min <= FLOAT_LIMIT: t_min = t_max * 100 
        
        target_xmax = t_max * 0.1
        target_xmin = t_min * 1.5
        self.ax.set_xlim(target_xmin, target_xmax)

        valid_mask = np.isfinite(positions)
        if np.any(valid_mask):
            y_vals = positions[valid_mask]
            positive_vals = y_vals[y_vals > FLOAT_LIMIT]
            
            if len(positive_vals) > 0:
                y_min_data = np.min(positive_vals)
                y_max_data = np.max(positive_vals)
            else:
                y_min_data = FLOAT_LIMIT
                y_max_data = FLOAT_LIMIT * 100

            if y_max_data <= y_min_data * 1.000001: 
                y_max_data = y_min_data * 10
            
            margin_factor = 5.0 
            target_ymin = y_min_data / margin_factor
            target_ymax = y_max_data * margin_factor
            if target_ymin < FLOAT_LIMIT: target_ymin = FLOAT_LIMIT
            self.ax.set_ylim(target_ymin, target_ymax)
    def _update_hopscotch(self, raw_data):
        if raw_data is None or not isinstance(raw_data, dict): return
        u_m = raw_data.get('u_minus', 0); u_p = raw_data.get('u_plus', 0)
        if not (np.isfinite(u_m) and np.isfinite(u_p)): return
        self.artists["particle"].set_offsets(np.column_stack([[u_m], [u_p]]))
        
        self.trail_data_x.append(u_m)
        self.trail_data_y.append(u_p)
        self.artists["trail"].set_data(self.trail_data_x, self.trail_data_y)
        
        points_to_show = 500
        recent_x = list(self.trail_data_x)[-points_to_show:]
        recent_y = list(self.trail_data_y)[-points_to_show:]
        if recent_x and recent_y:
            min_x, max_x = min(recent_x), max(recent_x)
            min_y, max_y = min(recent_y), max(recent_y)
            margin = 0.5
            target_xlim = (min_x - margin, max_x + margin)
            target_ylim = (min_y - margin, max_y + margin)
            cur_xlim = self.ax.get_xlim(); cur_ylim = self.ax.get_ylim()
            need_update = False
            new_xlim = list(cur_xlim); new_ylim = list(cur_ylim)
            if target_xlim[0] < cur_xlim[0]: new_xlim[0] = target_xlim[0]; need_update = True
            if target_xlim[1] > cur_xlim[1]: new_xlim[1] = target_xlim[1]; need_update = True
            if target_ylim[0] < cur_ylim[0]: new_ylim[0] = target_ylim[0]; need_update = True
            if target_ylim[1] > cur_ylim[1]: new_ylim[1] = target_ylim[1]; need_update = True
            if u_m < new_xlim[0] or u_m > new_xlim[1] or u_p < new_ylim[0] or u_p > new_ylim[1]:
                 new_xlim = target_xlim; new_ylim = target_ylim; need_update = True
            if need_update:
                self.ax.set_xlim(new_xlim); self.ax.set_ylim(new_ylim)
        
        lbl_from = raw_data.get('lbl_from', '?').replace('Grav', '') 
        lbl_to   = raw_data.get('lbl_to', '?').replace('Grav', '')
        lbl_aux  = raw_data.get('lbl_aux', '?').replace('Grav', '')
        lbl_a = raw_data.get('lbl_a', '?')
        lbl_b   = raw_data.get('lbl_b', '?')
        lbl_c  = raw_data.get('lbl_c', '?')
        info_str = (f"Current State: u+ = {u_p:.4f} | u- = {u_m:.4f}\n" f"Active triangle: a: {lbl_a}, b: {lbl_b}, c: {lbl_c}\n" f"Movement: {lbl_from} -> {lbl_to} (Aux: {lbl_aux})" )
        self.hopscotch_info_text.set_text(info_str)
        self.hopscotch_info_text.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
        self.hopscotch_info_text.set_fontsize(8)
    
    def _format_vector(self, v):
        try: return f"[{', '.join(f'{x:.2f}' for x in v)}]"
        except: return str(v)

    def set_time(self, new_time):
        self.current_frame = new_time
        self.trail_data_x = [] 
        self.trail_data_y = []
        if self.artists["trail"]:
            self.artists["trail"].set_data([], [])
        self._internal_update(new_time, force_update=True)
        self.canvas.draw_idle()
    def set_speed(self, new_speed): self.time_vel = new_speed
    def play(self): self.is_running = True
    def pause(self): self.is_running = False

class GraphicDisplay(Graphic):
    """ Wrapper de Tkinter. """
    def __init__(self, root, data, init_time=0.0, wall_data=None, update=None,
                 lims=[None, None], title="", graph_title="", axis_titles=["", ""],
                 grid=False, plot_type="plot", color="black", time_vel=1,
                 border=False, width=0, height=0, on_close_callback=None,
                 on_amplify_callback=None, on_step_callback=None, fargs=None,
                 enable_trail=True,prob_data=None): 
        
        self.frame = tk.Frame(root, width=width, height=height, bd=1, relief="solid")
        
        self.title_bar = tk.Frame(self.frame, height=25, bg="#e0e0e0", relief="flat")
        self.lbl_title = tk.Label(self.title_bar, text=title, bg="#e0e0e0", font=("Arial", 9, "bold"))
        self.btn_close = tk.Button(self.title_bar, text="✕", command=self.close, bd=0, padx=5, bg="#ffdddd")
        self.btn_amplify = tk.Button(self.title_bar, text="□", command=self.amplify, bd=0, padx=5)

        super().__init__(self.frame, data, init_time, wall_data, update, lims, 
                         graph_title, axis_titles, grid, plot_type, color, 
                         20, time_vel, border, fargs, on_step_callback, enable_trail,prob_data=prob_data)

        self.on_close_callback = on_close_callback
        self.on_amplify_callback = on_amplify_callback
        self.toolbar = None 
        self._internal_pack()

    def _internal_pack(self):
        self.btn_close.pack(side="right", fill="y")
        self.btn_amplify.pack(side="right", fill="y")
        self.lbl_title.pack(side="left", padx=5)
        self.title_bar.pack(side="top", fill="x")
        self.pack()
    def grid(self, row=0, column=0, padx=0, pady=0, sticky=""):
        self.frame.grid(row=row, column=column, padx=padx, pady=pady, sticky=sticky)
    def close(self):
        if self.on_close_callback: self.on_close_callback(self)
    def amplify(self):
        if self.on_amplify_callback: self.on_amplify_callback(self)
    
    # --- CORRECCIÓN 3: Gestión robusta de la leyenda al Amplificar/Desamplificar ---
    def set_amplify_state(self, is_maximized):
        self.is_amplified = is_maximized
        
        if self.toolbar is None:
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
            self.toolbar.update()
            
        if is_maximized:
            self.btn_amplify.config(text="−")
            self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Mostrar Leyenda si hay datos
            if self.legend_handles:
                # Convertimos valores del diccionario a lista
                handles_list = list(self.legend_handles.values())
                
                self.ax.legend(handles=handles_list, 
                               loc='upper left', 
                               bbox_to_anchor=(1.02, 1), 
                               fontsize='small', 
                               frameon=False,
                               title="Legend")
                self.fig.tight_layout(rect=[0, 0, 0.85, 1])
        else:
            self.btn_amplify.config(text="□")
            self.toolbar.pack_forget()
            
            # Ocultar Leyenda explícitamente
            leg = self.ax.get_legend()
            if leg:
                leg.remove()
            
            self.fig.tight_layout(rect=[0, 0, 1, 1])
            
        self._internal_update(self.current_frame, force_update=True)
        # Importante: forzar repintado inmediato para que se borre la leyenda
        self.canvas.draw() 

# ... (Clase Form se mantiene igual) ...
class Form():
    """A form is a combination of a text and an entry box or checkbox."""
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
        self.label.pack(side="left",anchor="w",padx=10)
        if self.entry_type=="numbers":
            self.entry.pack(side="right",anchor="e",padx=10)
        elif self.entry_type=="bool":
            self.checkbox.pack(side="left",anchor="e",padx=10)
    def pack(self,anchor="center",side="left",padx=0,pady=0):
        self.frame.pack(anchor=anchor,side=side,padx=padx,pady=pady)
        self._internal_pack()
    def place(self,anchor="center",relx=0,rely=0):
        self.frame.place(anchor=anchor,relx=relx,rely=rely)
        self._internal_pack()
    def grid(self,row=0,column=0,padx=0,pady=0,sticky="n"):
        self.frame.grid(row=row,column=column,padx=padx,pady=pady,sticky=sticky)
        self._internal_pack()
    def grid_remove(self):
        self.frame.grid_remove()
    def get(self):
        if self.entry_type=="numbers": return self.entry.get()
        elif self.entry_type=="bool": return self.check_state.get()