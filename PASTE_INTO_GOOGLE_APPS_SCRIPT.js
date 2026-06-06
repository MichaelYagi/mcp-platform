// ============================================================
// COPY AND PASTE THIS ENTIRE FILE INTO Google Apps Script
// https://script.google.com — create a new project, delete
// any existing code, then paste everything below this block.
// ============================================================
//
// SETUP (one time):
// 1. Go to https://script.google.com and create a new project
// 2. Delete all existing code and paste this entire file
// 3. Replace <SECRET_KEY> below with a strong random secret (32+ characters).
//    Generate one in your terminal: python3 -c "import secrets; print(secrets.token_urlsafe(32))"
//    Anyone who knows this key can read your emails and calendar — treat it like a password.
// 4. Click Deploy > New deployment > Web app
//      Execute as: Me
//      Who has access: Anyone
//    Click Deploy and copy the Web App URL
// 5. In your .env file set:
//      GOOGLE_APPS_SCRIPT_URL=<Web App URL>?key=<your SECRET_KEY>
// 6. On subsequent edits: Deploy > Manage deployments > edit the existing deployment
//    (do not create a new one or the URL will change)

const SECRET_KEY = '<SECRET_KEY>';
const MAX_EMAILS = 8;

function doGet(e) {
  if (e.parameter.key !== SECRET_KEY) {
    return ContentService.createTextOutput(JSON.stringify({ error: 'Unauthorized' }))
        .setMimeType(ContentService.MimeType.JSON);
  }

  const type = e.parameter.type;
  let data;

  if (type === 'gmail') {
    const maxEmails = parseInt(e.parameter.max_emails) || MAX_EMAILS;
    data = getGmailData(maxEmails);
  } else if (type === 'gmail_recent') {
    const maxEmails = parseInt(e.parameter.max_emails) || MAX_EMAILS;
    data = getGmailRecent(maxEmails);
  } else if (type === 'gmail_search') {
    const query = e.parameter.query || '';
    const maxResults = parseInt(e.parameter.max_results) || 20;
    data = getGmailSearch(query, maxResults);
  } else if (type === 'gmail_message') {
    const id = e.parameter.id || '';
    data = getGmailMessage(id);
  } else if (type === 'gmail_check_replied') {
    const threadId = e.parameter.thread_id || '';
    const sinceHours = e.parameter.since_hours ? parseFloat(e.parameter.since_hours) : null;
    data = checkGmailReplied(threadId, sinceHours);
  } else if (type === 'calendar') {
    const offset = parseInt(e.parameter.offset) || 0;
    const days = parseInt(e.parameter.days) || 1;
    data = getCalendarData(offset, days);
  } else {
    data = { error: 'Unknown type' };
  }

  return ContentService.createTextOutput(JSON.stringify(data))
      .setMimeType(ContentService.MimeType.JSON);
}

function doPost(e) {
  let body;
  try {
    body = JSON.parse(e.postData.contents);
  } catch (_) {
    return ContentService.createTextOutput(JSON.stringify({ error: 'Invalid JSON' }))
        .setMimeType(ContentService.MimeType.JSON);
  }

  if (body.key !== SECRET_KEY) {
    return ContentService.createTextOutput(JSON.stringify({ error: 'Unauthorized' }))
        .setMimeType(ContentService.MimeType.JSON);
  }

  const type = body.type;
  let result;

  if (type === 'send_email') {
    result = sendEmail(body);
  } else if (type === 'reply_email') {
    result = replyToEmail(body);
  } else if (type === 'create_event') {
    result = createCalendarEvent(body);
  } else {
    result = { error: 'Unknown type' };
  }

  return ContentService.createTextOutput(JSON.stringify(result))
      .setMimeType(ContentService.MimeType.JSON);
}

function sendEmail(params) {
  if (!params.to || !params.subject || !params.body) {
    return { error: 'Missing required fields: to, subject, body' };
  }
  const options = {};
  if (params.cc)       options.cc = params.cc;
  if (params.bcc)      options.bcc = params.bcc;
  if (params.replyTo)  options.replyTo = params.replyTo;
  if (params.htmlBody) options.htmlBody = params.htmlBody;
  GmailApp.sendEmail(params.to, params.subject, params.body, options);
  return { success: true };
}

function createCalendarEvent(params) {
  if (!params.title || !params.start) {
    return { error: 'Missing required fields: title, start' };
  }
  const cal = CalendarApp.getDefaultCalendar();
  const startTime = new Date(params.start);
  const options = {};
  if (params.description) options.description = params.description;
  if (params.location)    options.location = params.location;
  if (params.guests)      options.guests = params.guests;  // comma-separated emails

  if (params.allDay) {
    const endTime = params.end ? new Date(params.end) : null;
    if (endTime) {
      cal.createAllDayEvent(params.title, startTime, endTime, options);
    } else {
      cal.createAllDayEvent(params.title, startTime, options);
    }
  } else {
    // default to 1 hour if no end time given
    const endTime = params.end ? new Date(params.end) : new Date(startTime.getTime() + 60 * 60 * 1000);
    cal.createEvent(params.title, startTime, endTime, options);
  }
  return { success: true };
}

function getGmailData(maxEmails) {
  maxEmails = maxEmails || MAX_EMAILS;
  const threads = GmailApp.search('in:inbox is:unread');
  const messages = threads.slice(0, maxEmails).map(thread => {
    const msg = thread.getMessages()[thread.getMessageCount() - 1];
    const rawFrom = msg.getFrom();
    const from = rawFrom.replace(/<[^>]*>/, '').trim().replace(/"/g, '') || rawFrom;
    const messageId = msg.getId();
    const threadId = thread.getId();
    const preview = (msg.getPlainBody() || '')
        .replace(/[-_=*]{4,}/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .substring(0, 150);
    return {
      from: from,
      subject: thread.getFirstMessageSubject(),
      date: Utilities.formatDate(msg.getDate(), Session.getScriptTimeZone(), 'MMM d'),
      id: messageId,
      link: 'https://mail.google.com/mail/u/0/#inbox/' + threadId,
      preview: preview
    };
  });

  return {
    unreadCount: threads.length,
    messages: messages
  };
}

function replyToEmail(params) {
  if (!params.message_id || !params.body) {
    return { error: 'Missing required fields: message_id, body' };
  }
  const msg = GmailApp.getMessageById(params.message_id);
  if (!msg) return { error: 'Message not found: ' + params.message_id };
  const replyOptions = {};
  if (params.cc) replyOptions.cc = params.cc;
  msg.reply(params.body, replyOptions);
  const subject = msg.getSubject();
  return {
    status: 'sent',
    to: msg.getFrom(),
    subject: subject.toLowerCase().startsWith('re:') ? subject : 'Re: ' + subject
  };
}

function getGmailRecent(maxEmails) {
  maxEmails = maxEmails || MAX_EMAILS;
  const threads = GmailApp.search('in:inbox');
  const messages = threads.slice(0, maxEmails).map(thread => {
    const msg = thread.getMessages()[thread.getMessageCount() - 1];
    const rawFrom = msg.getFrom();
    const from = rawFrom.replace(/<[^>]*>/, '').trim().replace(/"/g, '') || rawFrom;
    const messageId = msg.getId();
    const threadId = thread.getId();
    const preview = (msg.getPlainBody() || '')
        .replace(/[-_=*]{4,}/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .substring(0, 150);
    return {
      from: from,
      subject: thread.getFirstMessageSubject(),
      date: Utilities.formatDate(msg.getDate(), Session.getScriptTimeZone(), 'MMM d'),
      id: messageId,
      link: 'https://mail.google.com/mail/u/0/#inbox/' + threadId,
      preview: preview,
      unread: msg.isUnread()
    };
  });
  return { count: messages.length, messages: messages };
}

function getGmailSearch(query, maxResults) {
  maxResults = maxResults || 20;
  if (!query) return { error: 'query is required' };
  const threads = GmailApp.search(query, 0, maxResults);
  const messages = [];
  threads.forEach(function(thread) {
    thread.getMessages().forEach(function(msg) {
      if (messages.length >= maxResults) return;
      const rawFrom = msg.getFrom();
      const from = rawFrom.replace(/<[^>]*>/, '').trim().replace(/"/g, '') || rawFrom;
      const messageId = msg.getId();
      const threadId = thread.getId();
      const snippet = (msg.getPlainBody() || '')
          .replace(/[-_=*]{4,}/g, ' ')
          .replace(/\s+/g, ' ')
          .trim()
          .substring(0, 150);
      messages.push({
        id: messageId,
        thread_id: threadId,
        from: from,
        subject: msg.getSubject() || thread.getFirstMessageSubject(),
        date: Utilities.formatDate(msg.getDate(), Session.getScriptTimeZone(), 'MMM d'),
        snippet: snippet,
        unread: msg.isUnread(),
        link: 'https://mail.google.com/mail/u/0/#inbox/' + messageId
      });
    });
  });
  return { query: query, count: messages.length, messages: messages };
}

function getGmailMessage(id) {
  if (!id) return { error: 'id is required' };
  const msg = GmailApp.getMessageById(id);
  if (!msg) return { error: 'Message not found: ' + id };
  const rawFrom = msg.getFrom();
  const from = rawFrom.replace(/<[^>]*>/, '').trim().replace(/"/g, '') || rawFrom;
  const threadId = msg.getThread().getId();
  const body = msg.getPlainBody() || msg.getBody().replace(/<[^>]+>/g, '').trim();
  return {
    id: msg.getId(),
    from: from,
    to: msg.getTo(),
    cc: msg.getCc() || '',
    subject: msg.getSubject(),
    date: Utilities.formatDate(msg.getDate(), Session.getScriptTimeZone(), "EEE, MMM d yyyy h:mm a"),
    body: body,
    link: 'https://mail.google.com/mail/u/0/#inbox/' + threadId
  };
}

function checkGmailReplied(threadId, sinceHours) {
  if (!threadId) return { error: 'thread_id is required' };
  const thread = GmailApp.getThreadById(threadId);
  if (!thread) return { error: 'Thread not found: ' + threadId };
  const myEmail = Session.getEffectiveUser().getEmail();
  const cutoff = sinceHours ? new Date(Date.now() - sinceHours * 3600 * 1000) : null;
  const sentMessages = thread.getMessages().filter(function(msg) {
    if (!msg.getFrom().includes(myEmail)) return false;
    if (cutoff && msg.getDate() < cutoff) return false;
    return true;
  });
  const replied = sentMessages.length > 0;
  const lastMsg = replied ? sentMessages[sentMessages.length - 1] : null;
  const lastReplyAt = lastMsg
      ? Utilities.formatDate(lastMsg.getDate(), Session.getScriptTimeZone(), "yyyy-MM-dd'T'HH:mm:ssZ")
      : null;
  return {
    thread_id: threadId,
    replied: replied,
    reply_count: sentMessages.length,
    last_reply_at: lastReplyAt,
    since_hours: sinceHours || null
  };
}

function getCalendarData(offset, numDays) {
  const tz = Session.getScriptTimeZone();
  const now = new Date();
  const startOfRange = new Date(now.getFullYear(), now.getMonth(), now.getDate() + offset, 0, 0, 0);
  const endOfRange = new Date(now.getFullYear(), now.getMonth(), now.getDate() + offset + numDays - 1, 23, 59, 59);

  const allItems = [];
  CalendarApp.getAllCalendars().forEach(cal => {
    const calId = cal.getId();
    cal.getEvents(startOfRange, endOfRange).forEach(ev => allItems.push({ ev, calId }));
  });

  allItems.sort((a, b) => {
    if (a.ev.isAllDayEvent() !== b.ev.isAllDayEvent()) return a.ev.isAllDayEvent() ? -1 : 1;
    return a.ev.getStartTime() - b.ev.getStartTime();
  });

  const todayKey = Utilities.formatDate(now, tz, 'yyyyMMdd');
  const tomorrowKey = Utilities.formatDate(
      new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1), tz, 'yyyyMMdd'
  );

  const dateStr = numDays === 1
      ? Utilities.formatDate(startOfRange, tz, 'EEEE, MMMM d')
      : Utilities.formatDate(startOfRange, tz, 'MMM d') + ' – ' + Utilities.formatDate(endOfRange, tz, 'MMM d');

  const eventList = allItems.map(({ ev: event, calId }) => {
    const eventKey = Utilities.formatDate(event.getStartTime(), tz, 'yyyyMMdd');
    const day = eventKey === todayKey ? 'Today'
        : eventKey === tomorrowKey ? 'Tomorrow'
            : Utilities.formatDate(event.getStartTime(), tz, 'EEE MMM d');

    const timeStr = event.isAllDayEvent()
        ? 'All day'
        : Utilities.formatDate(event.getStartTime(), tz, 'h:mm a');

    const guests = event.getGuestList()
        .map(g => g.getName() || g.getEmail())
        .filter(Boolean)
        .join(', ');

    const eid = Utilities.base64EncodeWebSafe(
      event.getId().replace('@google.com', '') + ' ' + calId
    ).replace(/=+$/, '');

    return {
      title: event.getTitle(),
      time: numDays > 1 ? day + ' · ' + timeStr : timeStr,
      allDay: event.isAllDayEvent(),
      startDate: Utilities.formatDate(event.getStartTime(), tz, 'yyyy-MM-dd'),
      location: event.getLocation() || '',
      notes: (event.getDescription() || '').replace(/<[^>]+>/g, '').trim().substring(0, 200),
      organizer: (event.getCreators() || [])[0] || '',
      attendees: guests,
      eid: eid,
      calendarLink: 'https://calendar.google.com/calendar/event?eid=' + eid
    };
  });

  return {
    date: dateStr,
    events: eventList
  };
}
