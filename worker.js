export default {
  async fetch(request) {
    const url = new URL(request.url);
    
    // Backend API (Railway, Fly.io, etc.) का URL
    const backendURL = "https://web-production-4026.up.railway.app/get_mp4";
    
    // reCAPTCHA Secret Key (इसे Backend में भी verify करें)
    const recaptchaSecret = "6LcAy9wqAAAAAIDpw8ywJb85n6UvmVYWq87N5w4s";

    // YouTube URL को एक्सट्रैक्ट करें
    const videoURL = url.searchParams.get("url");
    const recaptchaResponse = url.searchParams.get("recaptcha");

    if (!videoURL || !recaptchaResponse) {
      return new Response(JSON.stringify({ error: "Invalid Request" }), { status: 400 });
    }

    // reCAPTCHA Verify करें
    const recaptchaVerify = await fetch("https://www.google.com/recaptcha/api/siteverify", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: `secret=${recaptchaSecret}&response=${recaptchaResponse}`
    });

    const recaptchaResult = await recaptchaVerify.json();

    if (!recaptchaResult.success) {
      return new Response(JSON.stringify({ error: "reCAPTCHA failed" }), { status: 403 });
    }

    // Backend से YouTube वीडियो URL लाएं
    const backendResponse = await fetch(`${backendURL}?url=${encodeURIComponent(videoURL)}`);
    const backendData = await backendResponse.json();

    return new Response(JSON.stringify(backendData), {
      headers: { "Content-Type": "application/json" }
    });
  }
};
