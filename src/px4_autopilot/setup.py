from setuptools import setup

package_name = 'px4_autopilot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fdcl',
    maintainer_email='fdcl@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'extgcu_fm_module = px4_autopilot.extgcu_fm_module:main'
            'extgcu_autopilot_module = px4_autopilot.extgcu_autopilot_module:main'
            'extgcu_wpt_manage_module = px4_autopilot.extgcu_wpt_manage_module:main'
            'extgcu_controller_module = px4_autopilot.extgcu_controller_module:main'
        ],
    },
)
