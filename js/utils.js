function parseHaplotypeString(rawText) {
  const trimmed = rawText.trim();
  if (!trimmed) return [];
  const tokens = trimmed.split(/[\s,;\t]+/);
  return tokens.filter(token => {
    if (token === '') return false;
    return !isNaN(token) || /^[0-9-]+$/.test(token);
  });
}
