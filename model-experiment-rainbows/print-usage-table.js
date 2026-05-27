#!/usr/bin/env node

const fs = require('fs');
const data = JSON.parse(fs.readFileSync('./usage-data.json', 'utf8'));

// ANSI color codes
const CYAN   = '\x1b[36m';
const YELLOW = '\x1b[33m';
const RESET  = '\x1b[0m';
const BORDER = '\x1b[90m'; // dark gray

function fmt(n) {
  return n.toLocaleString('en-US');
}

function fmtCost(n) {
  return '$' + n.toFixed(2);
}

function sessionName(id) {
  const prefix = '-Users-leilakaneda-Projects-';
  return id.startsWith(prefix) ? id.slice(prefix.length) : id;
}

const cols = [
  { header: 'Session',       width: 20, align: 'left'  },
  { header: 'Models',        width: 16, align: 'left'  },
  { header: 'Input',         width: 9,  align: 'right' },
  { header: 'Output',        width: 9,  align: 'right' },
  { header: 'Cache Create',  width: 13, align: 'right' },
  { header: 'Cache Read',    width: 12, align: 'right' },
  { header: 'Total Tokens',  width: 13, align: 'right' },
  { header: 'Cost (USD)',    width: 11, align: 'right' },
  { header: 'Last Activity', width: 13, align: 'right' },
];

function pad(str, width, align) {
  str = String(str);
  if (str.length >= width) return str.slice(0, width);
  return align === 'right'
    ? str.padStart(width)
    : str.padEnd(width);
}

function sep(char = '─') {
  return BORDER + cols.map(c => char.repeat(c.width + 2)).join('┼') + RESET;
}

function topBorder() {
  return BORDER + '┌' + cols.map(c => '─'.repeat(c.width + 2)).join('┬') + '┐' + RESET;
}

function midBorder() {
  return BORDER + '├' + cols.map(c => '─'.repeat(c.width + 2)).join('┼') + '┤' + RESET;
}

function botBorder() {
  return BORDER + '└' + cols.map(c => '─'.repeat(c.width + 2)).join('┴') + '┘' + RESET;
}

function row(cells, color = '') {
  return BORDER + '│' + RESET +
    cells.map((cell, i) =>
      ' ' + color + pad(cell, cols[i].width, cols[i].align) + RESET + ' ' + BORDER + '│' + RESET
    ).join('');
}

// Print header
console.log(topBorder());
console.log(row(cols.map(c => c.header), CYAN));
console.log(midBorder());

// Print rows
data.sessions.forEach((s, idx) => {
  const models = (s.modelsUsed || []).map(m => {
    // Shorten model names: claude-sonnet-4-6 -> sonnet-4-6, claude-haiku-4-5 -> haiku-4-5
    return '- ' + m.replace(/^claude-/, '');
  });

  const name = sessionName(s.sessionId);

  // First line of row
  const firstLine = [
    name,
    models[0] || '',
    fmt(s.inputTokens),
    fmt(s.outputTokens),
    fmt(s.cacheCreationTokens),
    fmt(s.cacheReadTokens),
    fmt(s.totalTokens),
    fmtCost(s.totalCost),
    s.lastActivity || '',
  ];
  console.log(row(firstLine));

  // Additional model lines
  for (let m = 1; m < models.length; m++) {
    const extraLine = ['', models[m], '', '', '', '', '', '', ''];
    console.log(row(extraLine));
  }

  if (idx < data.sessions.length - 1) {
    console.log(midBorder());
  }
});

// Totals row
const t = data.totals;
console.log(midBorder());
const totalsLine = [
  'Total', '',
  fmt(t.inputTokens),
  fmt(t.outputTokens),
  fmt(t.cacheCreationTokens),
  fmt(t.cacheReadTokens),
  fmt(t.totalTokens),
  fmtCost(t.totalCost),
  '',
];
console.log(row(totalsLine, YELLOW));
console.log(botBorder());
