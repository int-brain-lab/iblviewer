from setuptools import setup, find_packages
#import iblviewer.launcher

setup(
    name='iblviewer',
    version='2.4.6',
    description='An interactive GPU-accelerated 3D viewer based on VTK',
    url='https://github.com/int-brain-lab/iblviewer',
    author='Nicolas Antille',
    author_email='nicolas.antille@gmail.com',
    license='MIT',
    install_requires=['numpy',
                      'matplotlib',
                      'requests',
                      'pandas',
                      'pynrrd',
                      'trimesh',
                      'k3d',
                      'vtk>=9.0',
                      'ipywebrtc',
                      #'ibllib>=1.6.0', # This optional dependency is huge (100MB)!
                      'iblutil',
                      'vedo>=2021.0.3',
                      'ipyvtklink',
                      'PyQt5',
                      'pyqt-darktheme'
                      ],
    packages=find_packages(include=['iblviewer','iblviewer.*', 'iblviewer_examples', 'iblviewer_examples.*']),
    package_data={'iblviewer_assets':['iblviewer_assets/*']},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "iblviewer = iblviewer.launcher:main",
            "iblviewer-points-demo = iblviewer_examples.ibl_point_neurons:main",
            "iblviewer-probes-demo = iblviewer_examples.ibl_insertion_probes:main",
            "iblviewer-coverage-demo = iblviewer_examples.ibl_brain_coverage:main",
            "iblviewer-volume-mapping-demo = iblviewer_examples.ibl_volume_mapping:main",
            "iblviewer-brain-wide-map = iblviewer_examples.ibl_brain_wide_map:main",
            "iblviewer-human-brain-demo = iblviewer_examples.human_brain:main",
            "iblviewer-headless-render-demo = iblviewer_examples.headless_render:main"
        ]
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.8',
    ]
)
