"""Grounded spectator wording. Facts come from the referee; plans remain attributed to a model."""
import re
import shlex

SEAT_RGB = {'A': (47, 124, 255), 'B': (255, 122, 26)}
COLOR_NAMES = {'A': 'Blue', 'B': 'Orange'}


def names(players):
    out = {s: str(players.get(s) or s) for s in ('A', 'B')}
    if out['A'].casefold() == out['B'].casefold():
        out = {s: f'{n} · {COLOR_NAMES[s]}' for s, n in out.items()}
    return out


def referee_reason(text, players, sport):
    """Translate only referee text; don't rewrite arbitrary model speech as a fact."""
    text = str(text or '')
    if sport == 'tabletennis':
        m = re.fullmatch(r"([AB])'s shot hit its own half", text)
        if m:
            return f"{players[m[1]]}'s return bounced on its own side of the table"
    return re.sub(r'\b[AB]\b', lambda m: players.get(m[0], m[0]), text)


def command_activity(command):
    """Observable operation, never a claim that code worked or a tactical change succeeded."""
    try:
        lexer = shlex.shlex(str(command), posix=True, punctuation_chars=';&|')
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    groups, group = [], []
    for token in tokens + [';']:
        if token and all(c in ';&|' for c in token):
            if group:
                groups.append(group)
            group = []
        else:
            group.append(token)
    found = []
    for args in groups:
        # Common launch wrappers; only recognize commands at the start of a shell segment.
        while args and (args[0] in ('timeout', 'nice', 'env', 'nohup') or re.match(r'^[A-Za-z_][\w]*=', args[0])):
            wrapper = args.pop(0)
            if wrapper in ('timeout', 'nice'):
                while args and args[0].startswith('-'):
                    args.pop(0)
                if args:
                    args.pop(0)
            elif wrapper == 'env':
                while args and '=' in args[0]:
                    args.pop(0)
        if not args:
            continue
        cmd = args[0].rsplit('/', 1)[-1]
        rest = args[1:]
        if cmd == 'game' and rest:
            act = {'ready': 'Reporting ready for the next point', 'observe': 'Checking the ball and robot positions',
                   'status': 'Checking the score and match state', 'camera': 'Inspecting the court view',
                   'review': 'Submitting a review of the last point', 'submit': 'Submitting the control program'}.get(rest[0])
            if act:
                found.append((3, act))
        elif cmd in ('python', 'python3', 'uv'):
            if '-m' in rest and 'pytest' in rest:
                found.append((2, 'Testing the control program'))
            elif '-m' in rest and 'py_compile' in rest:
                found.append((2, 'Checking the control program for syntax errors'))
            else:
                found.append((2, 'Running a control script'))
        elif cmd in ('cat', 'tee') and ('<<' in str(command) or '>' in str(command)):
            found.append((2, 'Writing the control program'))
        elif cmd in ('tail', 'cat', 'head', 'grep', 'rg'):
            found.append((1, "Checking the control program's output"))
        elif cmd in ('pytest',):
            found.append((2, 'Testing the control program'))
    return max(found, key=lambda item: item[0])[1] if found else None


def excerpt(text, limit=150):
    """Keep whole clauses when possible; never silently splice half a command into a sentence."""
    text = re.sub(r'\s+', ' ', str(text or '')).strip()
    if len(text) <= limit:
        return text
    clauses = re.split(r'(?<=[.!?。！？;；])\s*', text)
    out = ''
    for clause in clauses:
        if len(out) + len(clause) + 1 > limit:
            break
        out = (out + ' ' + clause).strip()
    if out:
        return out + ' …'
    cut = text[:limit].rsplit(' ', 1)[0]
    return (cut if len(cut) > limit // 2 else text[:limit]) + '…'


def caption_lines(caption):
    """Two explicitly attributed lines for a review, rather than an unexplained WHY→PLAN fragment."""
    kind, text = caption.get('kind', ''), caption.get('text', '')
    if kind == 'Review':
        summary = caption.get('summary')
        if summary:
            return [label + summary[key] for key, label in [('assessment', 'Assessment: '), ('plan', 'Next attempt: ')] if summary[key]]
        why, sep, plan = text.partition(' → ')
        if sep:
            return ['Assessment: ' + excerpt(why, 115), 'Next attempt: ' + excerpt(plan, 115)]
        return ['Model review: ' + excerpt(text, 180)]
    return [excerpt(text, 220)]
