from setuptools import setup, find_packages

setup(
    name = 'iblviewer', 
    version='1.0.2',   
    description='An interactive GPU-accelerated neuroscience viewer based on VTK that works with Jupyter Notebooks for the International Brain Laboratory.',
    url='https://github.com/int-brain-lab/iblviewer',
    author='Nicolas Antille',
    author_email='nicolas.antille@gmail.com',
    license='MIT',
    packages=['iblviewer'],
    install_requires=['numpy',
                      'matplotlib',
                      'pandas',
                      'pynrrd',
                      'trimesh',
                      'k3d',
                      'vtk>=9.0',
                      'ibllib>=1.6.0',
                      'vedo>=2021.0.2',
                      'ipyvtk-simple',
                      ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)