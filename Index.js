const express = require('express');
const ytdl = require('ytdl-core');
const app = express();
const port = process.env.PORT || 3000;

app.get('/download', async (req, res) => {
    const videoURL = req.query.url;
    if (!ytdl.validateURL(videoURL)) {
        return res.status(400).send('Invalid YouTube URL');
    }

    try {
        const info = await ytdl.getInfo(videoURL);
        const format = ytdl.chooseFormat(info.formats, { quality: '18' });
        res.json({ downloadLink: format.url });
    } catch (error) {
        res.status(500).send('Error downloading video');
    }
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
