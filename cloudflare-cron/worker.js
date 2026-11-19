// Pings the app every 10 minutes so the Render service never sleeps.
// /health also triggers a background data refresh if the data is older than REFRESH_INTERVAL_MIN.
export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(fetch(env.APP_URL + "/health", { headers: { "User-Agent": "keepalive-cron" } }));
  },
};