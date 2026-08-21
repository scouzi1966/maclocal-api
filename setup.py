from setuptools import Distribution, setup
from wheel.bdist_wheel import bdist_wheel


class NativeDistribution(Distribution):
    """Ensure wheels containing the AFM executable receive a platform tag."""

    def has_ext_modules(self) -> bool:
        return True


class AFMWheel(bdist_wheel):
    """Tag AFM's executable payload without tying it to one Python ABI."""

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        return ("py3", "none", "macosx_26_0_arm64")


setup(
    distclass=NativeDistribution,
    cmdclass={"bdist_wheel": AFMWheel},
)
