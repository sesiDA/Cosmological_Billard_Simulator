import matplotlib.pyplot as plt
import numpy as np

def plot_poincare_geometry():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Rango de coordenadas
    r = np.linspace(0, 2, 50)       # Radio para el cono
    theta = np.linspace(0, 2*np.pi, 50)
    R, THETA = np.meshgrid(r, theta)

    # 1. EL CONO DE LUZ (Beta_0^2 = Beta_1^2 + Beta_2^2)
    # Parametrización: x = r cos(t), y = r sin(t), z = r
    X_cone = R * np.cos(THETA)
    Y_cone = R * np.sin(THETA)
    Z_cone = R 
    
    # Dibujamos el cono como "wireframe" (alambre) para ver a través
    ax.plot_wireframe(X_cone, Y_cone, Z_cone, color='gray', alpha=0.3, rstride=5, cstride=5, label='Cono de Luz')

    # 2. EL HIPERBOLOIDE (Beta_0 = sqrt(1 + Beta_1^2 + Beta_2^2))
    # Parametrización usando funciones hiperbólicas para la hoja superior
    # Usamos rho como "radio hiperbólico"
    rho = np.linspace(0, 1.3, 30) 
    RHO, THETA_H = np.meshgrid(rho, theta)
    
    # x = sinh(rho) cos(theta)
    # y = sinh(rho) sin(theta)
    # z = cosh(rho)
    X_hyp = np.sinh(RHO) * np.cos(THETA_H)
    Y_hyp = np.sinh(RHO) * np.sin(THETA_H)
    Z_hyp = np.cosh(RHO)

    ax.plot_surface(X_hyp, Y_hyp, Z_hyp, color='cyan', alpha=0.6, edgecolor='blue', lw=0.5, label='Hiperboloide (M)')

    # 3. EL DISCO DE POINCARÉ (Proyección en z=0)
    # Dibujamos el disco unitario en el suelo
    r_disk = np.linspace(0, 1, 20)
    R_disk, THETA_D = np.meshgrid(r_disk, theta)
    X_disk = R_disk * np.cos(THETA_D)
    Y_disk = R_disk * np.sin(THETA_D)
    Z_disk = np.zeros_like(X_disk) # En el plano z=0

    ax.plot_surface(X_disk, Y_disk, Z_disk, color='orange', alpha=0.5, label='Disco Poincaré')
    
    # Dibuja el borde del disco
    theta_line = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta_line), np.sin(theta_line), 0, color='red', lw=2)

    # 4. LÍNEAS DE PROYECCIÓN (Visualización estereográfica)
    # Punto en el hiperboloide
    p_rho = 0.8
    p_theta = np.pi / 4
    ph_x = np.sinh(p_rho) * np.cos(p_theta)
    ph_y = np.sinh(p_rho) * np.sin(p_theta)
    ph_z = np.cosh(p_rho)
    
    # Proyección estereográfica al disco (Fórmula: u = x / (1+z))
    # NOTA: La proyección estereográfica suele ser desde el polo sur (-1, 0, 0) hacia el plano z=0.
    # Aquí visualizamos la correspondencia:
    pd_x = ph_x / (1 + ph_z)
    pd_y = ph_y / (1 + ph_z)
    
    # Dibujamos el punto en el hiperboloide y en el disco
    ax.scatter([ph_x], [ph_y], [ph_z], color='black', s=50, label='Punto P (Minkowski)')
    ax.scatter([pd_x], [pd_y], [0], color='red', s=50, label="Proyección P' (Poincaré)")
    
    # Dibujamos la línea que conecta (Proyección estereográfica pasa por el polo virtual en z=-1)
    ax.plot([ph_x, 0], [ph_y, 0], [ph_z, -1], color='black', linestyle='--', lw=1)
    # Extensión hasta el disco
    ax.plot([0, pd_x], [0, pd_y], [-1, 0], color='black', linestyle='--', lw=1)


    # Estética
    ax.set_xlabel(r'$\beta_2 /x$')
    ax.set_ylabel(r'$\beta_3 /y$')
    ax.set_zlabel(r'$\beta_1$ (Tiempo/Vol)')
    ax.set_title('Espacio de DeWitt: Cono, Hiperboloide y Poincaré')
    ax.view_init(elev=20, azim=45) # Ángulo de cámara
    
    # Límites para que se vea bonito
    ax.set_zlim(0, 2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    plt.show()

# Ejecutar la función
plot_poincare_geometry()