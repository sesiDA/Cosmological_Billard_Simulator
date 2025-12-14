import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


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
        if plot_type == "multiplot":
            # Límites dinámicos o iniciales (puedes ajustarlos luego)
            self.ax.set_xscale("log")
            self.ax.set_xlim(1e-12,1)  # Eje X (Tiempo) en log, de pequeño a grande
            
            # ACTIVAR ESCALA LOGARÍTMICA
            self.ax.set_yscale("log")
            self.ax.set_ylim(1e-1, 1)
            # Invertir eje X si quieres ver t -> 0 hacia la derecha
        else:
            self.set_lims(lims[0], lims[1])
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
        
        if not self.is_running:
            artists = []
            if hasattr(self, 'lines'): artists.extend(self.lines)
            if hasattr(self, 'line'): artists.append(self.line)
            return [a for a in artists if a is not None]

        # 2. Recuperar datos
        # Nota: get_history_data acepta *args, así que esto funciona para ambos casos
        
        print(f"DEBUG GRAPHIC: Update frame {frame}")
        data = self.update_func(frame, *args) if callable(self.update_func) else None
        
        # 3. Callback de tiempo (solo si la simulación avanza)
        if self.on_step_callback:
            self.on_step_callback(self.current_frame)
            
        if data is not None:
            
            # --- CASO A: Gráfico de Series Temporales (Multiplot) ---
            # Aquí data[0] es un ARRAY de tiempos. Sí podemos usar len().
            if hasattr(self, 'lines'):
                # Verificamos que sea un array/lista y que tenga datos
                if len(data) >= 2 and hasattr(data[0], '__len__') and len(data[0]) > 0:
                    times = data[0]
                    positions = data[1]
                    for i, line in enumerate(self.lines):
                        if i < positions.shape[1]:
                            line.set_data(times, positions[:, i])
                    try:
                        valid_pos = positions[positions > 0]
                        if len(valid_pos) > 0:
                            y_min = np.min(valid_pos)
                            y_max = np.max(valid_pos)
                            
                            # Obtenemos los límites actuales para ver si hay que expandir
                            current_ylim = self.ax.get_ylim()
                            
                            # Expandimos solo si los datos se salen (con un margen del 10% en log)
                            updated = False
                            new_ymin, new_ymax = current_ylim
                            
                            if y_min < current_ylim[0]:
                                new_ymin = y_min * 0.1 # Margen inferior
                                updated = True
                            if y_max > current_ylim[1]:
                                new_ymax = y_max * 10.0 # Margen superior
                                updated = True
                        current_t = times[-1]
                        
                        if current_t > 1e-300: # Protección numérica
                            # get_xlim devuelve (izq, dcha). Al estar invertido: (1.0, limite_pequeño)
                            current_xlim = self.ax.get_xlim()
                            limit_right = current_xlim[1] 
                            
                            # Si el tiempo actual ha superado (es menor que) el borde derecho...
                            if current_t < limit_right:
                                # ...Estiramos el eje un orden de magnitud más
                                new_right = current_t * 0.1
                                self.ax.set_xlim(1.0, new_right) # Mantenemos el 1.0 fijo a la izquierda
                                updated = True        
                        if updated:
                            self.ax.set_ylim(new_ymin, new_ymax)
                    except Exception:
                        pass # Evitar crash si hay datos raros
                    current_t = times[-1]
                    self.ax.set_xlim(1.0, 0.0)
                    return self.lines
            # --- CASO B: Gráfico Scatter/Plot Normal ---
            # Aquí data[0] es un NÚMERO (float). NO usamos len().
            elif hasattr(self, 'line'):
                if hasattr(self.line, 'set_offsets'): # Scatter
                    # set_offsets espera una matriz [[x, y]]
                    self.line.set_offsets(np.column_stack([data[0], data[1]]))
                else: # Plot
                    self.line.set_data(data[0], data[1])
                return self.line,
            
        # Retorno de seguridad (lista de artistas)
        artists = []
        if hasattr(self, 'lines'): artists.extend(self.lines)
        if hasattr(self, 'line'): artists.append(self.line)
        return [a for a in artists if a is not None]
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
        elif plot_type=="multiplot":
            self.lines = []
            if data is None or len(data[0]) == 0:
                return
            times = data[0]
            positions = data[1]
            n_dims = positions.shape[1]    
            
            colors = plt.cm.jet(np.linspace(0, 1, n_dims))
            for i in range(n_dims):
                line, = self.ax.plot(times, positions[:, i], label=f"Dim {i+1}", color=colors[i])
                self.lines.append(line)
            self.ax.legend(loc="upper right", fontsize='small')
            
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