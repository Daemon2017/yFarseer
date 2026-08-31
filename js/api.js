async function fetchPrediction(payload) {
  const url = `${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.PREDICT}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload),
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    if (!response.ok) {
      throw new Error(`Сервер ответил ошибкой: ${response.status} ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error('Время ожидания ответа от сервера (15 секунд) истекло.');
    }
    throw error;
  }
}
