from iblviewer.mouse_brain import MouseBrainViewer


def main():
    # This example starts the viewer and renders a view
    viewer = MouseBrainViewer()
    # Font size is made larger for 4K rendering
    viewer.initialize(resolution=50, mapping='Allen', embed_font_size=30, offscreen=True)
    # Put objects on the scene
    viewer.show()

    # Add more code to add data and control the viewer here

    # Render a 4K image.
    viewer.set_info_visibility(False)
    #viewer.render('./test.jpg', 3840, 2160)
    viewer.close()


if __name__ == '__main__':
    main()