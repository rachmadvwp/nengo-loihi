

class DebugN2Board(object):

    def __init__(self, n2board):
        self.n2board = n2board

    @classmethod
    def _indent(cls, lines):
        return [('- ' if i == 0 else '  ') + line
                for i, line in enumerate(lines)]

    @classmethod
    def board_str(cls, board):
        out_lines = []
        out_lines.append("N2Board(id=%s)" % board.id)

        for chip in board.n2Chips:
            lines = cls._indent(cls.chip_str(chip).split('\n'))
            out_lines.extend(lines)

        return '\n'.join(out_lines)

    @classmethod
    def chip_str(cls, chip):
        out_lines = []
        out_lines.append("N2Chip(id=%s)" % chip.id)

        for core in chip.n2Cores:
            lines = cls._indent(cls.core_str(core).split('\n'))
            out_lines.extend(lines)

        return '\n'.join(out_lines)

    @classmethod
    def core_str(cls, core):
        out_lines = []
        out_lines.append("N2Core(id=%s)" % core.id)
        return '\n'.join(out_lines)

    def __str__(self):
        return self.board_str(self.n2board)
