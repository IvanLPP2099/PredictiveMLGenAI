import streamlit as st
from paginas import login, dashboardDemo


def main():
    # Configurar la página y el estado de la sesión (solo una vez en app.py)
    st.set_page_config(
        page_title=":beginner: Dashboard Sales",  # Título de la página
        page_icon=":smile:",           # Icono de la página
        layout="wide",                 # Configurar el layout para que ocupe todo el ancho
        initial_sidebar_state="expanded"  # Barra lateral expandida por defecto
    )

    # Leer parámetros de la URL
    query_params = st.query_params
    logged_in = query_params.get("logged_in", ["False"])[0] == "True"
    
    # Verificar si el usuario está logueado
    if logged_in or ("logged_in" in st.session_state and st.session_state.get("logged_in", False)):
        st.session_state.logged_in = True  # Asegurar consistencia interna del estado
        if "archivo_subido" not in st.session_state:
            st.session_state.archivo_subido = False
        dashboardDemo.mostrar_dashboard()
    else:
        # Si no, mostramos el login
        login.showLogin()
        # Si inicia sesión correctamente, actualiza el parámetro en la URL
        if "logged_in" in st.session_state and st.session_state.logged_in:
            st.query_params.set(logged_in="True")

if __name__ == "__main__":
    main()