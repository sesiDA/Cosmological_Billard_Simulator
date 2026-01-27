from libraries.init_UI import init_UI
from libraries.user_interface import UserInterface

if __name__ == "__main__":
    
    """Main code"""
    
    init_form = init_UI("Simulation Initialization") #Starts form
    if not init_form.closed: #If form wasn't closed starts simulation
        ui=UserInterface("Cosmological Billard Simulation",init_form.parameters)
