import streamlit as st
import pandas as pd
import json 
from streamlit_lottie import st_lottie
import os
import time
from .userManagement import verifyCredentials

def validateCredentials(usuario, contrasena):
    return verifyCredentials(usuario, contrasena)

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
    
def showLogin():
    
    c1, c2 = st.columns([60, 40])
    
    with c1:
        # Ajusta la ruta al archivo JSON de animación
        ruta_animacion_laptop = os.path.join("animations", "laptopUser.json")
        lottie_coding = load_lottiefile(ruta_animacion_laptop)
        st_lottie(
            lottie_coding,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=None,
            width=None,
            key=None,
        )

    with c2:
        st.title("🔐  Inicio de :blue[Sesión] :sunglasses:")

        # Formulario de inicio de sesión
        with st.form("login_form"):
            usuario = st.text_input("Usuario 👇")
            contrasena = st.text_input("Contraseña 👇", type="password")
            boton_login = st.form_submit_button("Iniciar Sesión", type="primary",use_container_width=True)

        # Validación de credenciales
        if boton_login:
            if validateCredentials(usuario, contrasena):
                st.session_state.logged_in = True
                st.session_state.usuario = usuario
                aviso = st.success("Inicio de sesión exitoso. Redirigiendo al dashboard...")
                time.sleep(3)
                aviso.empty()
                # Simular redirección recargando el flujo principal
                st.session_state.pagina_actual = "dashboard"
                st.rerun()
            else:
                aviso = st.error("Usuario o contraseña incorrectos")
                time.sleep(3)
                aviso.empty()
