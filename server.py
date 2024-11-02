import eel


# Функция, которая будет вызвана из JavaScript
@eel.expose
def say_hello():
    print("приветики")


# Инициализация Eel
eel.init('web')
# Запуск приложения
eel.start('index.html', size=(500, 500))
