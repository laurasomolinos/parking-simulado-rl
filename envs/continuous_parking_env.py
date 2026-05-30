import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class ContinuousParkingEnv(gym.Env):
    """
    Entorno simplificado para asistencia al aparcamiento.
    El agente controla un vehículo en un mundo 2D continuo.
    Debe aparcar en una plaza objetivo evitando colisiones con obstáculos.
    Principalmente tiene que tener las funciones: render(), reset(), step() y close() 
    Para ser compatible con Gymnasium ademas de funciones que complementen a estas
    """

    metadata = {"render_modes": ["human"], "render_fps": 10} #en modo humano, muestra grafico por pantalla 
    #es necesario para Gymnasium 

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        # Dimensiones del mundo en metros
        self.world_width = 10.0
        self.world_height = 6.0

        self.previous_action = None
        self.best_distance_to_goal = None
        self.stagnation_steps = 0

        # Parámetros de simulación
        self.dt = 0.1  # lo que avanza cada acción en segundos 
        self.max_steps = 300 #limite pasos por episodio
        self.max_speed = 0.5 #valocidad max del vehiculo m/s
        self.max_turn_rate = 1.2 #velocidad de giro rad/s (angular) maxima 

        # Sensores -> forman una distribución angular alrededir del vehiculo 
        #por ello dividimos el "circulo" de alrededor del vehiculo en 8 partes, cada una con un sensor 
        self.max_sensor_range = 2.5 #si no encuentra obstaculo en 2.5m devuelve esa distancia 
        self.sensor_angles = np.array([
            0.0, # frontal
            np.pi / 4, # frontal izq
            -np.pi / 4, # frontal drcha
            np.pi / 2, # izq
            -np.pi / 2, # drcha
            np.pi, # detras
            3 * np.pi / 4, # detras izq
            -3 * np.pi / 4 # detras drcha
        ], dtype=np.float32)

        # Acciones discretas: hemos decidido recoger 7 posibilidades 
        # 0 avanzar recto
        # 1 avanzar girando izquierda
        # 2 avanzar girando derecha
        # 3 retroceder recto
        # 4 retroceder girando izquierda
        # 5 retroceder girando derecha
        # 6 frenar / detenerse
        self.action_space = spaces.Discrete(7)

        # Observación continua: estado que el agente recibe en cada paso 
        # x_norm, y_norm, cos(theta), sin(theta), v_norm,
        # 8 sensores,
        # distancia_objetivo_norm,
        # cos(angulo_objetivo), sin(angulo_objetivo),
        # cos(error_orientacion), sin(error_orientacion)]
        # en total 18 valores en rango [-1, 1]
        #usamos seno y coseno en vez de theta porque la posterior red neuronal aprendera mejor
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(18,),
            dtype=np.float32
        )

        # Plaza de aparcamiento
        self.parking_center = np.array([8.5, 3.0], dtype=np.float32) #ahí ponemos la posicion objetivo 
        self.parking_theta = np.pi / 2  # orientación objetivo (mirando hacia arriba)

        # Obstáculos rectangulares: (x, y, w, h)
        # x,y indican la esquina inferior izquierda
        self.obstacles = [(7.0, 4.2, 1.2, 1.0),#simula coche aparcado arriba
            (7.0, 0.8, 1.2, 1.0),#simula coche aparcado abajo
            (4.0, 2.5, 0.8, 1.0),#simula obstáculo lateral en el camino 
            ]

        #Estado interno continuo, se inicializa como nose y en rest() se asignan valores 
        self.x = None #coordenada x actual
        self.y = None #coordenada y actual
        self.theta = None #orientacion actual en radianes
        self.v = None #velocidad actual en m/s
        self.steps = 0 #contador de pasos en el episodio actual
        self.previous_distance_to_goal = None #distancia anterior para calcular recompensa
        self.previous_orientation_error = None #error de orientacion anterior tambien para recompensa 

        # Render
        self.fig = None #figura de matplotlib para renderizar el entorno la creamos en render()
        self.ax = None #ejes de matplotlib para renderizar el entorno, tambien se crean en render()

    def reset(self, seed=None, options=None):
        """Reinicia el entorno a un estado inicial fijo."""
        super().reset(seed=seed)
        #Estado inicial fijo para empezar luego podremos aleatorizarlo.
        self.x = 1.0
        self.y = 3.0
        self.theta = 0.0
        self.v = 0.0
        self.steps = 0
        self.previous_action = None
        self.best_distance_to_goal = self._distance_to_goal()
        self.stagnation_steps = 0
        self.previous_distance_to_goal = self._distance_to_goal()
        self.previous_orientation_error = abs(self._orientation_error())
        obs = self._get_obs() #observacion inicial del entorno, se calcula a partir del estado interno y se devuelve al agente
        info = self._get_info(collision=False, parked=False) 
        #info inicial del entorno, se calcula a partir del estado interno y se devuelve al agente, incluye informacion adicional como distancia a objetivo, si ha colisionado, ...
        return obs, info
    
    """Funciones para ejecutar luego el step(action)"""
    def _action_to_control(self, action):
        """Convierte la acción discreta en comandos que seran las decisiones que el agente tomará"""
        action_table = {0: (1.0, 0.0), # avanzar recto
            1: (1.0, 1.0), # avanzar izquierda
            2: (1.0, -1.0),# avanzar derecha
            3: (-1.0, 0.0),# retroceder recto
            4: (-1.0, 1.0), # retroceder izquierda
            5: (-1.0, -1.0), # retroceder derecha
            6: (0.0, 0.0), # frenar
            }
        return action_table[int(action)]
    
    def _cast_ray(self, angle):
        """Lanza un rayo en la dirección dada y devuelve la distancia al primer obstáculo encontrado."""
        step_size = 0.1 #paso de avance del rayo, si se disminuye es mas preciso pero mas lento 
        distance = 0.0 #distancia inicial del rayo, se va incrementando hasta encontrar obstaculo

        while distance <= self.max_sensor_range: #mientras no se alcance el rango maximo del sensor
            px = self.x + distance * np.cos(angle) #calculamos la posicion del rayo a partir de la posicion actual del vehiculo
            py = self.y + distance * np.sin(angle)

            #Si el rayo sale del mundo, consideramos que ha encontrado un obstáculo a esa distancia
            if px < 0.0 or px > self.world_width or py < 0.0 or py > self.world_height:
                return distance

            #Comprobar colisión con obstáculos
            for ox, oy, ow, oh in self.obstacles:
                if ox <= px <= ox + ow and oy <= py <= oy + oh:
                    return distance
            distance += step_size
        return self.max_sensor_range
    
    def _compute_sensors(self):
        """
        Simula sensores de distancia usando el ray cas.
        Devuelve distancias normalizadas: 1.0 = no hay obstáculo cerca, 0.0 = obstáculo pegado
        """
        distances = []
        for relative_angle in self.sensor_angles: #calculamos el ángulo absoluto del rayo sumando el ángulo relativo del sensor al ángulo actual del vehículo
            ray_angle = self.theta + relative_angle
            distance = self._cast_ray(ray_angle) #lanzamos el rayo y obtenemos la distancia al obstáculo más cercano en esa dirección
            distances.append(distance / self.max_sensor_range) #normalizamos la distancia dividiéndola por el rango máximo del sensor para que esté entre 0 y 1

        return np.array(distances, dtype=np.float32) #devolvemos las distancias de los sensores como un array

    def _get_obs(self):
        """Calcula la observación actual a partir del estado interno del entorno."""
        sensors = self._compute_sensors() #obtenemos las distancias de los sensores 
        x_norm = 2.0 * (self.x / self.world_width) - 1.0 #normalizamos las coordenadas para que esté entre -1 y 1
        y_norm = 2.0 * (self.y / self.world_height) - 1.0 
        v_norm = np.clip(self.v / self.max_speed, -1.0, 1.0) #normalizamos la velocidad actual dividiéndola por la velocidad máxima y limitándola entre -1 y 1
        distance_to_goal = self._distance_to_goal() 
        max_distance = np.sqrt(self.world_width ** 2 + self.world_height ** 2) #distancia máxima posible en el mundo, se usa para normalizar la distancia al objetivo
        distance_norm = np.clip(distance_to_goal / max_distance, 0.0, 1.0)# normalizamos igual la distancia pero entre 0 y 1 
        angle_to_goal = self._angle_to_goal()
        orientation_error = self._orientation_error()
        obs = np.array([x_norm, #acumulamos toda la informacion en un array de observación que se devuelve al agente
            y_norm,
            np.cos(self.theta),
            np.sin(self.theta),
            v_norm,
            *sensors,
            distance_norm,
            np.cos(angle_to_goal),
            np.sin(angle_to_goal),
            np.cos(orientation_error),
            np.sin(orientation_error),
        ], dtype=np.float32) 

        return obs

    def _get_info(self, collision, parked):
        sensors = self._compute_sensors() 
        return { #informacion adicional que se devuelve al agente en cada paso, puede ser útil para análisis o posteriormente 
            "x": float(self.x),
            "y": float(self.y),
            "theta": float(self.theta),
            "v": float(self.v),
            "distance_to_goal": float(self._distance_to_goal()),
            "orientation_error": float(self._orientation_error()),
            "min_sensor_distance": float(np.min(sensors)),
            "collision": bool(collision),
            "is_success": bool(parked),
            "steps": int(self.steps),
        }

    def _compute_reward(
        self,
        action,
        old_distance,
        new_distance,
        old_orientation_error,
        new_orientation_error,
        min_sensor_distance,
        collision,
        parked
    ):
        """Lejos de la plaza → lo importante es acercarse.
        Cerca de la plaza → lo importante es orientar bien.
        Cuando está cerca y orientado → frenar es bueno.
        Frenar lejos → malo.
        Oscilar → malo.
        Chocar → muy malo."""
        reward = -0.08

        # Progreso hacia la plaza
        delta_distance = old_distance - new_distance
        reward += 4.0 * delta_distance

        # Si no hay progreso real, pequeña penalización
        if abs(delta_distance) < 0.003:
            reward -= 0.03

        # Orientación: más importante cerca del objetivo
        delta_orientation = old_orientation_error - new_orientation_error

        if new_distance < 1.0:
            reward += 3.0 * delta_orientation
        else:
            reward += 0.5 * delta_orientation

        # Penalización por riesgo
        if min_sensor_distance < 0.15:
            reward -= 4.0
        elif min_sensor_distance < 0.30:
            reward -= 1.0

        # Penalización por oscilación
        reverse_pairs = {
            (0, 3), (3, 0),
            (1, 5), (5, 1),
            (2, 4), (4, 2),
        }

        if self.previous_action is not None:
            if (self.previous_action, action) in reverse_pairs:
                reward -= 0.7

        # Frenar lejos está mal, pero frenar cerca y alineado está bien
        if action == 6:
            if new_distance < 0.45 and new_orientation_error < 0.40:
                reward += 5.0
            else:
                reward -= 1.0

        # Colisión
        if collision:
            reward -= 120.0

        # Bonus progresivo por estar muy cerca del centro de la plaza
        if new_distance < 0.45:
            reward += 2.0 * (0.45 - new_distance)

        # Aparcamiento correcto
        if parked:
            reward += 150.0

        return float(reward)

    def _distance_to_goal(self):
        pos = np.array([self.x, self.y], dtype=np.float32)
        return float(np.linalg.norm(pos - self.parking_center))

    def _angle_to_goal(self):
        dx = self.parking_center[0] - self.x #diferencia en x e y entre la posición actual y el centro de la plaza 
        dy = self.parking_center[1] - self.y
        angle = np.arctan2(dy, dx) - self.theta #ángulo absoluto al objetivo restando la orientación actual del vehículo 
        resul = (angle + np.pi) % (2 * np.pi) - np.pi #normalizamos el ángulo para que esté entre -pi y pi
        return resul

    def _orientation_error(self):
        #diferencia entre la orientación actual del vehículo y la orientación objetivo de la plaza
        angle = self.parking_theta - self.theta
        return (angle + np.pi) % (2 * np.pi) - np.pi #normalizamos el error de orientación para que esté entre -pi y pi
    
    def _is_parked(self, action):
        """para considerar que el vehículo ha aparcado correctamente,
        debe estar cerca del objetivo, con la orientación correcta, 
        casi sin velocidad y haber ejecutado la acción de frenar"""
        distance = self._distance_to_goal()
        orientation_error = abs(self._orientation_error())
        return (distance < 0.35
            and orientation_error < 0.25
            and abs(self.v) < 0.05
            and action == 6)

    def _check_collision(self):
        # Colisión con límites del mundo
        if self.x < 0.0 or self.x > self.world_width:
            return True
        if self.y < 0.0 or self.y > self.world_height:
            return True
        # Colisión con obstáculos
        for ox, oy, ow, oh in self.obstacles:
            if ox <= self.x <= ox + ow and oy <= self.y <= oy + oh:
                return True

        return False

    def step(self, action):
        """Ejecuta la acción dada y actualiza el estado del entorno."""
        self.steps += 1
        old_distance = self._distance_to_goal() #distancia anterior a objetivo para calcular recompensa
        old_orientation_error = abs(self._orientation_error()) #error de orientacion anterior para calcular recompensa
        #Aplicar acción
        speed_cmd, turn_cmd = self._action_to_control(action) 

        self.v = speed_cmd * self.max_speed
        omega = turn_cmd * self.max_turn_rate

        # Actualizar orientación
        #calculamos el nuevo ángulo a partir del ángulo actual y la velocidad de giro omega multiplicada por el tiempo que dura la acción dt
        angle = self.theta + omega * self.dt 
        self.theta = (angle + np.pi) % (2 * np.pi) - np.pi #normalizamos el ángulo para que esté siempre entre -pi y pi

        # Actualizar posición
        self.x += self.v * np.cos(self.theta) * self.dt #calculamos la nueva posición a partir de la posición actual, la velocida, la orientacion y el tiempo 
        self.y += self.v * np.sin(self.theta) * self.dt

        # Comprobar colisión y aparcamiento
        collision = self._check_collision()
        parked = self._is_parked(action)

        new_distance = self._distance_to_goal() #nueva distancia a objetivo para calcular recompensa
        if new_distance < self.best_distance_to_goal - 0.01:
            self.best_distance_to_goal = new_distance
            self.stagnation_steps = 0
        else:
            self.stagnation_steps += 1
        #actualizamos el resto de valores 
        new_orientation_error = abs(self._orientation_error())
        min_sensor_distance = np.min(self._compute_sensors())

        # Calcular recompensa
        reward = self._compute_reward(action=action,
            old_distance=old_distance,
            new_distance=new_distance,
            old_orientation_error=old_orientation_error,
            new_orientation_error=new_orientation_error,
            min_sensor_distance=min_sensor_distance,
            collision=collision,
            parked=parked)

        terminated = collision or parked
        # Si durante 80 pasos no mejora su mejor distancia al objetivo, se corta el episodio y se penaliza.
        truncated = self.steps >= self.max_steps or self.stagnation_steps >= 80
        if self.stagnation_steps >= 80:
            reward -= 20.0
        obs = self._get_obs()
        info = self._get_info(collision=collision, parked=parked)
        self.previous_action = int(action)
        return obs, reward, terminated, truncated, info

    def render(self):
        """Renderiza el entorno usando Matplotlib. Solo se muestra si render_mode es human """
        if self.render_mode != "human":
            return
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 5)) #creamos la figura y los ejes de matplotlib para renderizar el entorno
            #solo se hace la primera vez que se llama a render() 

        self.ax.clear() #limpiamos los ejes para dibujar el nuevo estado del entorno en cada paso

        # Mundo
        self.ax.set_xlim(0, self.world_width) #establecemos los límites de los ejes 
        self.ax.set_ylim(0, self.world_height)
        self.ax.set_aspect("equal") #para que las unidades sean iguales en ambos ejes
        self.ax.set_title("Entorno Parking Simulado")#título del gráfico

        # Obstáculos
        for ox, oy, ow, oh in self.obstacles:
            rect = plt.Rectangle((ox, oy), ow, oh, alpha=0.6) #dibujamos los coches aparcados
            self.ax.add_patch(rect)

        # Plaza de aparcamiento
        parking_rect = plt.Rectangle(
            (self.parking_center[0] - 0.4, self.parking_center[1] - 0.6), 0.8, 1.2,
            fill=False, linewidth=2)
        self.ax.add_patch(parking_rect) #dibujamos la plaza de aparcamiento como un rectángulo vacío

        # Vehículo como punto + dirección
        self.ax.plot(self.x, self.y, marker="o", markersize=10)
        self.ax.arrow(self.x, self.y, 0.4 * np.cos(self.theta), 0.4 * np.sin(self.theta),
            head_width=0.15, length_includes_head=True) #dibujamos el vehículo como un punto con una flecha que indica su orientación

        #Sensores
        for relative_angle in self.sensor_angles:
            ray_angle = self.theta + relative_angle
            d = self._cast_ray(ray_angle)
            self.ax.plot([self.x, self.x + d * np.cos(ray_angle)],
                [self.y, self.y + d * np.sin(ray_angle)],
                linestyle="--",
                linewidth=0.8) #dibujamos los rayos de los sensores como líneas discontinuas
        plt.pause(0.01) #pausa para actualizar el gráfico, para controlar la velocidad de renderizado

    def close(self): # cerramos la figura de matplotlib cuando se cierra el entorno
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


if __name__ == "__main__":
    #Prueba rápida del entorno con acciones aleatorias
    env = ContinuousParkingEnv(render_mode="human")

    obs, info = env.reset()

    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        if terminated or truncated:
            print("Episodio terminado")
            break

    env.close()