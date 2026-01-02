from libraries.init_UI import init_UI
from libraries.user_interface import UserInterface

if __name__ == "__main__":
    
    """Main code"""
    
    init_form = init_UI("Simulation Initialization") #Starts form
    if not init_form.closed: #If form wasn't closed starts simulation
        ui=UserInterface("Cosmological Billard Simulation",init_form.parameters)
        
#TODO   
    #Clase Graphic
        #Etiquetas dinamicas de los graficos
    #clase init_UI  
        #Opciones de visualización del billar.(Canonica, plano de geodesica comovil, plano geodesica comovil inicial)
        #Opciones de parametrización del eje de kasner
        #Opciones de visualizar/no visualizar ciertos graficos
        #Load Simulation
    #clase SimConfig    
        #File:
            #Save Simulation as...
            #Load Simulation
        #Edit
            #Add Slice
            #Add 3D Slice
            #Add Simulation
            #Delete Simulation
            #Modify Simulation conditions
            #Modify Slice contditions
            #Change Position
            #Change Velocity
            #Change Matter Content
            #Set/Unset symetry walls
        #Export
            #Export Simulation data as (.csv,.txt)
            #Export video(.mp4)
        #View
            #View SimInfo pannel
            #View spacial curvature
            #View particle stela
            #View Weyl Chamber zone
            #View all walls/just dominant
            #View probability density in slice
            #Change slice plane recalibration at bounce
    #Crear ejecutable .exe para reducir riesgo de error.
    #Comentar codigo