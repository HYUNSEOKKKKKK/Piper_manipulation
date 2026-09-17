from glob import glob

from setuptools import setup

package_name = 'piper_pnp'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'LICENSE']),
        ('share/' + package_name + '/urdf', glob('urdf/*')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Piper_manipulation contributors',
    maintainer_email='125255496+HYUNSEOKKKKKK@users.noreply.github.com',
    description='RGB-D cuboid manipulation with guarded PiPER control and MoveIt 2.',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'piper_pnp_controller = piper_pnp.piper_pnp_controller:main',
            'fake_aruco_publisher = piper_pnp.fake_aruco:main',
            'piper_pnp_estop = piper_pnp.estop:main',
        ],
    },
)
