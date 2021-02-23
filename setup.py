from setuptools import setup, find_packages

setup(
    name = 'iblviewer', 
    version='1.0.0',   
    description='A example Python package',
    url='https://github.com/int-brain-lab/iblviewer',
    author='Nicolas Antille',
    author_email='nicolas.antille@gmail.com',
    license='MIT',
    packages=['iblviewer'],
    install_requires=['numpy',
                      'vtk>=9.0',
                      'ibllib>=1.5.37',
                      'matplotlib',
                      'pandas',
                      'pynrrd',
                      'trimesh',
                      'vedo',
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