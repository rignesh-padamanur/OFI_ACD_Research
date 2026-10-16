"""Check platform flags; numerical likelihood checks live in test_chronological_acd."""
from pathlib import Path
import unittest
from unittest.mock import patch

from src.chronological_acd import kernel_build_command


class KernelPortabilityTests(unittest.TestCase):
    def command(self, platform, environment=None):
        with patch('src.chronological_acd.sys.platform', platform), \
             patch.dict('src.chronological_acd.os.environ', environment or {}, clear=True), \
             patch('src.chronological_acd.shutil.which', return_value='/usr/bin/cc'):
            return kernel_build_command(Path('kernel.c'), Path('kernel.so'))

    def test_macos_preserves_historical_compiler_flags(self):
        command, compiler = self.command('darwin')
        self.assertEqual(command, ['/usr/bin/clang', '-dynamiclib', '-O3', '-std=c99',
                                   '-ffp-contract=off', 'kernel.c', '-o', 'kernel.so'])
        self.assertEqual(compiler, ['/usr/bin/clang'])

    def test_linux_builds_shared_position_independent_object(self):
        command, compiler = self.command('linux')
        self.assertEqual(command, ['cc', '-shared', '-fPIC', '-O3', '-std=c99',
                                   '-ffp-contract=off', 'kernel.c', '-o', 'kernel.so', '-lm'])
        self.assertEqual(compiler, ['cc'])

    def test_compiler_override_preserves_arguments_without_shell_expansion(self):
        command, compiler = self.command('linux', {'CC': 'clang -fno-fast-math'})
        self.assertEqual(compiler, ['clang', '-fno-fast-math'])
        self.assertEqual(command[:2], compiler)

    def test_windows_has_explicit_unsupported_platform_message(self):
        with self.assertRaisesRegex(RuntimeError, 'Windows.*WSL'):
            self.command('win32')

    def test_missing_compiler_has_actionable_error(self):
        with patch('src.chronological_acd.sys.platform', 'linux'), \
             patch.dict('src.chronological_acd.os.environ', {}, clear=True), \
             patch('src.chronological_acd.shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'C99 compiler'):
                kernel_build_command(Path('kernel.c'), Path('kernel.so'))


if __name__ == '__main__':
    unittest.main()
