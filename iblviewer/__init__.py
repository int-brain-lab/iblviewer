try:
    import iblviewer.mouse_brain
    import iblviewer.qt_application
except ModuleNotFoundError:
    pass
import iblviewer.launcher
import iblviewer.collection
import iblviewer.objects
import iblviewer.utils
import iblviewer.volume
import iblviewer.application

if __name__ == '__main__':
    app = iblviewer.launcher.main()