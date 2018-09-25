

class DebugN2Board(object):

    def __init__(self, n2board):
        self.n2board = n2board

    @classmethod
    def _indent(lines):
        if len(lines) > 0:
            lines[0].insert(0, '- ')
        for line in lines[1:]:
            line.insert(0, '  ')
        return lines

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
        out_lines.append("N2Core(id=%s)" % chip.id)
        return '\n'.join(out_lines)

    def __str__(self):
        return self.board_str(self.n2board)
