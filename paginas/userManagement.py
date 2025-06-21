import MySQLdb
from .conexionMysql import get_db_connection

def verifyCredentials(username: str, password: str) -> bool:
    """
    Verifica si las credenciales del usuario son válidas.
    Retorna True si el usuario existe y la contraseña es correcta.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT COUNT(*) FROM usuarios WHERE username = %s AND password = %s"
            cursor.execute(query, (username, password))
            resultado = cursor.fetchone()
            return resultado[0] > 0
    except MySQLdb.Error as e:
        print(f"Error en la verificación de credenciales: {e}")
        return False

def getDataUser(correo: str) -> dict | None:
    """
    Devuelve un diccionario con los datos del usuario si existe, o None si no se encuentra.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(MySQLdb.cursors.DictCursor)
            query = "SELECT nombre, apellido, correo, telefono FROM usuarios WHERE correo = %s"
            cursor.execute(query, (correo,))
            return cursor.fetchone()
    except MySQLdb.Error as e:
        print(f"Error al obtener datos del usuario: {e}")
        return None
