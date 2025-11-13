import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ttkbootstrap as tb
import tkinter as tk
from tkinter import ttk
from matplotlib.animation import FuncAnimation

# ===============================
# DISTRIBUTIONS
# ===============================
DIST_TYPES = ["Gaussienne", "Uniforme", "Cercle", "Anneau", "Spirale"]

def generate_point_cloud(dist_type, params, n_points):
    if dist_type == "Gaussienne":
        mu_x, mu_y, sigma = params
        return np.random.multivariate_normal([mu_x, mu_y], [[sigma,0],[0,sigma]], n_points)
    elif dist_type == "Uniforme":
        min_x, max_x, min_y, max_y = params
        return np.column_stack((np.random.uniform(min_x, max_x, n_points),
                                np.random.uniform(min_y, max_y, n_points)))
    elif dist_type == "Cercle":
        cx, cy, r = params
        angles = np.random.uniform(0, 2*np.pi, n_points)
        radii = r * np.sqrt(np.random.uniform(0,1,n_points))
        return np.column_stack((cx + radii*np.cos(angles), cy + radii*np.sin(angles)))
    elif dist_type == "Anneau":
        cx, cy, r_inner, r_outer = params
        angles = np.random.uniform(0,2*np.pi,n_points)
        radii = np.sqrt(np.random.uniform(r_inner**2, r_outer**2, n_points))
        return np.column_stack((cx + radii*np.cos(angles), cy + radii*np.sin(angles)))
    elif dist_type == "Spirale":
        cx, cy, b = params
        theta = np.linspace(0, 4*np.pi, n_points)
        r = b * theta
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        return np.column_stack((x, y))

def generate_points(class_params):
    X = []
    np.random.seed(0)
    for params in class_params:
        dist_type, n_points, p = params
        X.append(generate_point_cloud(dist_type, p, int(n_points)))
    return np.vstack(X)

# ===============================
# K-MEANS++
# ===============================
def initialize_centroids_kmeanspp(X, k):
    n_samples = X.shape[0]
    centroids = [X[np.random.randint(0,n_samples)]]
    for _ in range(1,k):
        distances = np.min(np.linalg.norm(X[:,np.newaxis]-np.array(centroids),axis=2),axis=1)
        probs = distances**2 / np.sum(distances**2)
        centroids.append(X[np.random.choice(n_samples,p=probs)])
    return np.array(centroids)

def initialize_centroids_random(X,k):
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def kmeans_steps(X,k,method,max_iter=10):
    centroids = initialize_centroids_kmeanspp(X,k) if method=="K-Means++" else initialize_centroids_random(X,k)
    for _ in range(max_iter):
        distances = np.linalg.norm(X[:,np.newaxis]-centroids,axis=2)
        labels = np.argmin(distances,axis=1)
        yield labels, centroids
        new_centroids = np.array([X[labels==i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids,new_centroids,atol=1e-4): break
        centroids = new_centroids

def kmeans_visual_animated(X,k,ax,canvas,method,max_iter=10,progress_bar=None,status_label=None):
    colors = plt.cm.get_cmap("tab10",k)
    steps = list(kmeans_steps(X,k,method,max_iter))

    padding_factor = 2
    x_std, y_std = np.std(X[:,0]), np.std(X[:,1])
    x_min, x_max = X[:,0].min()-padding_factor*x_std, X[:,0].max()+padding_factor*x_std
    y_min, y_max = X[:,1].min()-padding_factor*y_std, X[:,1].max()+padding_factor*y_std
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,400), np.linspace(y_min,y_max,400))
    grid = np.c_[xx.ravel(), yy.ravel()]

    def update(frame):
        ax.clear()
        labels, centroids = steps[frame]
        distances_grid = np.linalg.norm(grid[:,np.newaxis]-centroids,axis=2)
        Z = np.argmin(distances_grid,axis=1).reshape(xx.shape)
        ax.contourf(xx,yy,Z,alpha=0.2,levels=np.arange(-0.5,k,1),cmap="tab10")
        for i in range(k):
            ax.scatter(X[labels==i,0],X[labels==i,1],color=colors(i),label=f'Cluster {i+1}',alpha=0.7)
            ax.scatter(centroids[i,0],centroids[i,1],color='black',marker='x',s=100,linewidths=3)
            ax.text(centroids[i,0],centroids[i,1],f"C{i+1}", fontsize=10, fontweight='bold',
                    color='black', ha='center', va='center')
        ax.set_title(f"Itération {frame+1}/{len(steps)} ({method})")
        ax.grid(True); ax.axis("equal"); ax.legend()
        if progress_bar: progress_bar['value']=(frame+1)/len(steps)*100
        if status_label: status_label.config(text=f"Iteration {frame+1}/{len(steps)}")

    ani = FuncAnimation(fig,update,frames=len(steps),interval=600,repeat=False)
    canvas.draw()

# ===============================
# INTERFACE
# ===============================
def update_class_table(*args):
    for widget in frame_classes.winfo_children():
        widget.destroy()
    n_classes = int(entry_classes.get())
    headers = ["Classe","Distribution","Paramètres","Points"]
    for col, h in enumerate(headers):
        ttk.Label(frame_classes, text=h, font=("Segoe UI", 10, "bold")).grid(row=0, column=col, padx=5, pady=2)
    global class_entries
    class_entries = []
    max_columns = 4

    for i in range(n_classes):
        ttk.Label(frame_classes, text=f"{i+1}").grid(row=i+1, column=0, sticky="n")
        dist_menu = ttk.Combobox(frame_classes, values=DIST_TYPES, width=12)
        dist_menu.current(0)
        dist_menu.grid(row=i+1, column=1, sticky="n")
        param_frame = ttk.Frame(frame_classes)
        param_frame.grid(row=i+1, column=2, sticky="n")

        def create_params(dist_name, frame):
            for w in frame.winfo_children():
                w.destroy()
            if dist_name == "Gaussienne":
                param_names, defaults = ["μx","μy","σ"], [5,0,1]
            elif dist_name == "Uniforme":
                param_names, defaults = ["min_x","max_x","min_y","max_y"], [-5,5,-5,5]
            elif dist_name == "Cercle":
                param_names, defaults = ["cx","cy","r"], [0,0,5]
            elif dist_name == "Anneau":
                param_names, defaults = ["cx","cy","r_inner","r_outer"], [0,0,3,5]
            elif dist_name == "Spirale":
                param_names, defaults = ["cx","cy","b"], [0,0,0.5]
            spinboxes = []
            for col, name, default in zip(range(max_columns), param_names+[None]*(max_columns-len(param_names)), defaults+[0]*(max_columns-len(defaults))):
                if name:
                    sb = ttk.Spinbox(frame, from_=-50, to=50, increment=0.5, width=6)
                    sb.set(default)
                    sb.grid(row=1, column=col, sticky="n", padx=2, pady=0.1)
                    spinboxes.append(sb)
            return spinboxes

        spinboxes = create_params("Gaussienne", param_frame)
        points = ttk.Spinbox(frame_classes, from_=10, to=1000, increment=10, width=6)
        points.set(100)
        points.grid(row=i+1, column=3, sticky="n")
        class_entries.append((dist_menu, param_frame, *spinboxes, points))

        def update_params(event, idx=i):
            dist_name = class_entries[idx][0].get()
            spinboxes_new = create_params(dist_name, class_entries[idx][1])
            class_entries[idx] = tuple([class_entries[idx][0], class_entries[idx][1], *spinboxes_new, class_entries[idx][-1]])

        dist_menu.bind("<<ComboboxSelected>>", update_params)

def display_clouds():
    ax.clear()
    class_params=[]
    for e in class_entries:
        dist_type = e[0].get()
        points = int(e[-1].get())
        param_widgets = e[2:-1]
        params=[]
        for w in param_widgets:
            params.append(float(w.get()))
        class_params.append((dist_type, points, params))
    X = generate_points(class_params)
    ax.scatter(X[:,0],X[:,1],c='gray',s=20)
    ax.set_title("Nuages de points générés")
    ax.grid(True)
    ax.axis("equal")
    canvas.draw()

def launch_kmeans():
    ax.clear()
    n_classes = int(entry_classes.get())
    method = init_method.get()
    class_params=[]
    for e in class_entries:
        dist_type = e[0].get()
        points = int(e[-1].get())
        param_widgets = e[2:-1]
        params=[]
        for w in param_widgets:
            params.append(float(w.get()))
        class_params.append((dist_type, points, params))
    X = generate_points(class_params)
    ax.scatter(X[:,0],X[:,1],c='gray',s=20)
    ax.set_title("Données initiales")
    ax.grid(True)
    ax.axis("equal")
    canvas.draw()
    progress_bar.start()
    kmeans_visual_animated(X,k=n_classes,ax=ax,canvas=canvas,method=method,max_iter=10,
                           progress_bar=progress_bar,status_label=status_label)
    progress_bar.stop()
    status_label.config(text="Convergence terminée")

# ===============================
# FENETRE PRINCIPALE
# ===============================
root = tb.Window(themename="cosmo")
root.title("K-Means Visualizer Interactif")
root.geometry("1400x850")

# --- Styles pour titres en gras ---
style = ttk.Style()
style.configure("Bold.TLabelframe.Label", font=("Ubuntu", 12, "bold"))  # plus grand

# --- Banner ---
banner = ttk.Label(root, text="🔹 K-Means Visualizer 🔹", font=("Segoe UI", 22, "bold"), foreground="#2c3e50")
banner.pack(pady=10)

frame_controls = ttk.Frame(root)
frame_controls.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

# --- Bloc Bienvenue et instructions ---
frame_desc_general = ttk.Labelframe(frame_controls, text="Bienvenue et instructions", padding=10, style="Bold.TLabelframe")
frame_desc_general.pack(pady=(10,15), fill="x")

welcome_text = tk.Text(frame_desc_general, wrap="word", bg="#f8f8f8", fg="#333", font=("Segoe UI", 10), height=10, relief="solid", bd=1)
welcome_text.pack(fill="x")

welcome_text.tag_configure("title", font=("Segoe UI", 10, "bold"), foreground="#2c3e50")
welcome_text.tag_configure("bold", font=("Segoe UI", 10, "bold"))

welcome_text.insert("end", "👋 Bienvenue dans le ", ())
welcome_text.insert("end", "K-Means Visualizer Interactif", ("bold",))
welcome_text.insert("end", " !\n\n", ())

welcome_text.insert("end", "💡 Instructions générales :\n", ("title",))
welcome_text.insert("end", " \n", ())
welcome_text.insert("end",
    "1. Sélectionnez le nombre de classes et la méthode d'initialisation pour K-Means.\n"
    "2. Configurez les paramètres des nuages de points selon la distribution choisie.\n"
    "3. Cliquez sur ", ())
welcome_text.insert("end", "'Afficher Nuages'", ("bold",))
welcome_text.insert("end", " pour visualiser les points générés.\n")
welcome_text.insert("end", "4. Cliquez sur ", ())
welcome_text.insert("end", "'Lancer K-Means'", ("bold",))
welcome_text.insert("end", " pour voir l'algorithme s'exécuter étape par étape.\n\n")

welcome_text.insert("end",
    "Vous pouvez choisir parmi plusieurs distributions : ", ())
welcome_text.insert("end", "Gaussienne, Uniforme, Cercle, Anneau, Spirale", ("bold",))
welcome_text.insert("end", ", et ajuster leurs paramètres pour explorer la formation des clusters.", ())

welcome_text.config(state="disabled")

# --- Bloc Nombre de classes et Méthode côte à côte ---
frame_top_controls = ttk.Frame(frame_controls)
frame_top_controls.pack(pady=5, fill="x")

ttk.Label(frame_top_controls, text="Nombre de classes :", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w", padx=(0,5))
entry_classes = ttk.Spinbox(frame_top_controls, from_=1, to=10, increment=1, width=5)
entry_classes.set(2)
entry_classes.grid(row=0, column=1, sticky="w", padx=(0,30))
entry_classes.bind("<Return>", lambda e: update_class_table())
entry_classes.bind("<FocusOut>", lambda e: update_class_table())

ttk.Label(frame_top_controls, text="Méthode d'initialisation :", font=("Segoe UI", 11, "bold")).grid(row=0, column=2, sticky="w", padx=(0,5))
init_method = ttk.Combobox(frame_top_controls, values=["Aléatoire","K-Means++"])
init_method.current(1)
init_method.grid(row=0, column=3, sticky="w")


# --- Description des distributions ---
frame_desc_distrib = ttk.Labelframe(frame_controls, text="Description des distributions", padding=10, style="Bold.TLabelframe")
frame_desc_distrib.pack(pady=(10,15), fill="x")

desc_text = tk.Text(frame_desc_distrib, wrap="word", bg="#f8f8f8", fg="#333", font=("Segoe UI", 10), height=10, relief="solid", bd=1)
desc_text.pack(fill="x")

desc_text.tag_configure("title", font=("Segoe UI", 10, "bold"), foreground="#2c3e50")
desc_text.tag_configure("bold", font=("Segoe UI", 10, "bold"))

desc_text.insert("end", "📘 Voici les types de distributions disponibles :\n\n", ("title",))

desc_text.insert("end", "• Gaussienne : ", ("bold",))
desc_text.insert("end", "génère un nuage de points suivant une loi normale centrée sur (μx, μy) avec un écart-type σ.\n")

desc_text.insert("end", "• Uniforme : ", ("bold",))
desc_text.insert("end", "répartit les points de manière uniforme dans le rectangle défini par (min_x, max_x, min_y, max_y).\n")

desc_text.insert("end", "• Cercle : ", ("bold",))
desc_text.insert("end", "crée des points uniformément répartis à l’intérieur d’un cercle de rayon r et de centre (cx, cy).\n")

desc_text.insert("end", "• Anneau : ", ("bold",))
desc_text.insert("end", "génère des points entre deux rayons (r_inner, r_outer) autour d’un centre (cx, cy).\n")

desc_text.insert("end", "• Spirale : ", ("bold",))
desc_text.insert("end", "dispose les points le long d’une spirale paramétrée par b et centrée en (cx, cy).\n")

desc_text.config(state="disabled")


# --- Paramètres des nuages ---
frame_classes = ttk.Labelframe(frame_controls, text="Paramètres des nuages", padding=10, style="Bold.TLabelframe")
frame_classes.pack(pady=(10,15), fill="x")
update_class_table() 

# --- Boutons côte à côte ---
frame_buttons = ttk.Frame(frame_controls)
frame_buttons.pack(pady=(0,20))
ttk.Button(frame_buttons, text="Afficher Nuages", command=display_clouds).grid(row=0, column=0, padx=5)
ttk.Button(frame_buttons, text="Lancer K-Means", command=launch_kmeans).grid(row=0, column=1, padx=5)

# --- Barre de progression en bas ---
progress_bar = ttk.Progressbar(frame_controls, mode='determinate', length=300)
progress_bar.pack(pady=(10,5))
status_label = ttk.Label(frame_controls, text="En attente...", font=("Segoe UI", 10, "italic"))
status_label.pack(pady=(0,10))

# --- Zone de tracé ---
frame_plot = ttk.Frame(root)
frame_plot.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

plt.style.use('seaborn-v0_8-muted')
fig, ax = plt.subplots(figsize=(8,8))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f0f0f0")
canvas = FigureCanvasTkAgg(fig, master=frame_plot)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

root.mainloop()
