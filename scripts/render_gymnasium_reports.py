#!/usr/bin/env python3
"""Refresh result tables and the overview while preserving detailed report layouts and media."""
import json
import os
from html import escape
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'outputs/reports/gym-adapt/gymnasium'
ENVIRONMENTS = ('inverted_pendulum', 'air_raid', 'ant', 'half_cheetah', 'hopper', 'walker2d', 'humanoid')
OTHER = (('pendulum', 'Pendulum'), ('lunar_lander', 'Lunar Lander'), ('swimmer', 'Swimmer'), ('asterix', 'Asterix'))
COLUMNS = (('profile', 'Profile mean / P95 (ms)'), ('l0_teacher', 'L0 teacher → L0'), ('l0_vla', 'L0 VLA → L0'), ('baseline', 'L0 VLA → real-time (B)'), ('profile_teacher', 'Profile teacher → profile (E10)'), ('final', 'Profile VLA → real-time (F)'))
METHOD = ('Returns are mean ± population standard deviation across episodes. The profile column reports latency mean / P95 in milliseconds. '
          'L0 is simulated zero latency; B/F use measured real-time inference latency. E10 is the ten-episode simulated profile-teacher evaluation, '
          'not selection20; repeated selection seeds are disclosed per model. Source L0 teachers retain their stated clocks. '
          'A dash means unavailable or not run; the model status explains which. One training run and separately realized delays do not establish '
          'controlled architecture rankings or equal-latency causal effects. Teacher data gates are not VLA acceptance thresholds.')


def relative(path, directory):
    return os.path.relpath(path, directory)


def source_url(url, directory):
    return url if url.startswith('https://') else relative(ROOT / url, directory)


def link(label, url):
    return f'<a href="{escape(url, quote=True)}">{escape(label)}</a>'


def markdown_table(rows, overview=False):
    headers = (['Environment'] if overview else []) + ['Model'] + [label for _, label in COLUMNS]
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for env, model in rows:
        values = ([f'[{env["title"]}]({env["environment"]}/index.html)'] if overview else []) + [model['name'] + ' · ' + model['status']]
        values += [model['metrics'][key] for key, _ in COLUMNS]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def page(title, subtitle, navigation, content):
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{escape(title)} · Latency-sensitive bench</title><link rel="stylesheet" href="inverted_pendulum/styles.css"><link rel="stylesheet" href="report-layout.css"></head>
<body class="report-unified"><a class="skip-link" href="#main">Skip to results</a><header class="hero"><div class="hero-inner"><p class="eyebrow">Latency-sensitive bench / Experiment reports</p><h1>{escape(title)}</h1><p class="lede">{escape(subtitle)}</p></div></header><div class="layout"><aside class="toc" aria-label="Report navigation"><a class="toc-title" href="#main">Gymnasium</a><nav>{navigation}</nav></aside><main id="main">{content}</main></div><footer>Latency-sensitive bench · Gymnasium experiment reports</footer></body></html>
'''


def gain(env, model):
    metrics = model['metrics']
    if metrics['final'] == '—':
        return '—'
    directory = REPORTS / env['environment']
    if model['id'] == 'Pi05':
        name = 'pi05-results.json' if env['environment'] in ('inverted_pendulum', 'air_raid') else 'pi05-paired-results.json'
        paired = json.loads((directory / name).read_text())
        return f'{paired["mean_gain"]:+,.2f}<small>{paired["gain_percent"]:+,.2f}%</small>'
    if model['id'] == 'GR00T':
        results = json.loads((directory / 'results-summary.json').read_text())['GR00T']
        baseline, final = results['B']['mean_return'], results['F']['mean_return']
    else:
        paired = json.loads((directory / 'openvla-paired-results.json').read_text())
        baseline, final = paired['B_mean'], paired['F_mean']
    delta = final - baseline
    return f'{delta:+,.2f}<small>{delta / baseline * 100:+,.2f}%</small>'


def results_table(env, *, overview=False):
    models = {model['id']: model for model in env['models']}
    rows = []
    for key in ('Pi05', 'GR00T', 'OpenVLA'):
        model = models[key]
        target = ('./' + env['environment'] + '/index.html' if overview else '') + '#' + key.lower()
        rows.append('<tr><th scope="row">' + link(model['name'], target) + '<small>' + escape(model['status']) + '</small></th><td>' + escape(model['metrics']['baseline']) + '</td><td>' + escape(model['metrics']['final']) + '</td><td>' + gain(env, model) + '</td></tr>')
    return '<div class="table-wrap" role="region" tabindex="0" aria-label="Measured real-time baseline and profile-model results"><table class="results-table"><caption>Measured real-time evaluation · mean ± population SD · protocols in each model section</caption><thead><tr><th scope="col">Model</th><th scope="col">B · zero-trained</th><th scope="col">F · profile-trained</th><th scope="col">Mean gain</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'


def final_text(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [sentence for sentence in sentences if not re.search(r'\brevision\b|superseded native.*H8|old GR00T H8', sentence, re.I)]
    return ' '.join(sentences).replace('historical shared source teacher', 'shared source teacher').replace('Historical source clock', 'Source clock')


def render_environment(env):
    directory = REPORTS / env['environment']
    path = directory / 'index.html'
    html = path.read_text()
    # Detailed narratives and figures are curated; refresh only the summary table.
    overview = html.index('<section id="overview">')
    start = html.index('<div class="table-wrap"', overview)
    end = html.index('</div>', html.index('</table>', start)) + len('</div>')
    path.write_text(html[:start] + results_table(env) + html[end:])
    md = [f'# {env["title"]}', '[HTML report](index.html) · [All environments](../index.html) · [Results JSON](results.json)', markdown_table([(env, model) for model in env['models']]), METHOD]
    for model in env['models']:
        md += [f'## {model["name"]}', model['status']]
        for key, label in [('protocol', 'Protocol'), ('training', 'Training'), ('data', 'Data')]:
            md += [f'### {label}', final_text(model[key])]
        md += ['### Evaluation scope', '\n'.join('- ' + final_text(item) for item in model['limitations'] if not re.search('superseded.*H8', item, re.I))]
        md += ['### Models, data and evidence']
        for source in model['sources']:
            label = source['label'].replace('Original', '').replace('Preserved original', '').strip()
            md.append(f'- [{label}]({source_url(source["url"], directory)})')
    (directory / 'README.md').write_text('\n\n'.join(md) + '\n')


def render_overview(environments):
    nav = ''.join(link(env['title'], '#' + env['environment']) for env in environments)
    html = ''
    for env in environments:
        html += '<section id="' + env['environment'] + '"><p class="section-kicker">Results / final report</p><h2>' + link(env['title'], env['environment'] + '/index.html') + '</h2>' + results_table(env, overview=True) + '<p class="report-note">Each model retains its stated FPS, training recipe and evaluation protocol. A dash indicates an evaluation that was not run; scientific-stop status is retained.</p></section>'
    html += '<section><h2>Reading the results</h2><p>' + METHOD + '</p></section>'
    html += '<section><h2>Other adaptation reports</h2><ul>' + ''.join('<li>' + link(title, env + '/index.html') + '</li>' for env, title in OTHER) + '</ul></section>'
    (REPORTS / 'index.html').write_text(page('Gymnasium results', 'Seven environments · Pi0.5, GR00T and OpenVLA · measured real-time results and experimental evidence', nav, html))
    rows = [(env, model) for env in environments for model in env['models']]
    md = ['# Gymnasium results', '[HTML overview](index.html).', markdown_table(rows, True), METHOD, '## Report maintenance', 'Each environment’s results.json records the comparison metrics. Detailed narratives, protocols and figures are curated in its report. `python scripts/render_gymnasium_reports.py` refreshes the overview and result tables while preserving page layouts and media.']
    (REPORTS / 'README.md').write_text('\n\n'.join(md) + '\n')


if __name__ == '__main__':
    environments = [json.loads((REPORTS / env / 'results.json').read_text()) for env in ENVIRONMENTS]
    for environment in environments:
        if environment['environment'] != 'inverted_pendulum':
            render_environment(environment)
    render_overview(environments)
    print('Refreshed six detailed reports and the seven-environment overview; InvertedPendulum remains the approved style reference.')
