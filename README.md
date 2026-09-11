# IUST RAG — راهنمای راه‌اندازی و مرجع API (Phase 1)

سیستم RAG مرکز کامپیوتر دانشگاه علم و صنعت ایران  
- **Laravel**: Auth / RBAC / CRUD اسناد / تاریخچه چت / Gateway به Python  
- **Python**: Embedding / Hybrid Retrieval (Qdrant) / LLM / Ingest  
- **Qdrant**: Vector DB (dense + sparse)  
- **Redis**: Cache پاسخ RAG + صف Job  
- **MySQL**: Source of truth سازمانی  

| سرویس | آدرس پیش‌فرض |
|--------|----------------|
| Laravel | `http://127.0.0.1:8000` |
| Python | `http://127.0.0.1:8001` |
| Qdrant | `http://127.0.0.1:6333` |
| Redis | `127.0.0.1:6379` |

تمام routeهای Laravel زیر prefix استاندارد: **`/api`**.

---

## ۱. احراز هویت مشترک

| سناریو | هدر |
|--------|------|
| کاربر / فرانت → Laravel | `Authorization: Bearer {sanctum_token}` |
| Laravel → Python (ingest/delete/wipe/…) | `X-Internal-Key: {PYTHON_INTERNAL_API_KEY}` |
| فرانت مستقیم به Python (تست) | `Authorization: Bearer …` **یا** `user_context` در Body |

کلیدها باید یکسان باشند:

- Laravel: `PYTHON_INTERNAL_API_KEY`
- Python: `LARAVEL__INTERNAL_API_KEY`

---

## ۲. راه‌اندازی محلی

### 2.1 پیش‌نیاز

- PHP 8.2+ ، Composer  
- Python 3.11+ ، venv  
- MySQL (schema فعلی)  
- Redis Server  
- Qdrant  

### 2.2 Laravel

cd iust-rag-laravel
composer install
# تنظیم .env (DB، Redis، PYTHON_*)
php artisan key:generate
php artisan db:seed --class=RbacSeeder   # در صورت نیاز
php artisan serve --host=127.0.0.1 --port=8000

### Worker صف (ترمینال جدا — برای reembed):
php artisan queue:work redis --timeout=3600 --tries=1

### نمونه کلیدهای .env لاراول:
PYTHON_BASE_URL=http://127.0.0.1:8001
PYTHON_TIMEOUT=180
PYTHON_INGEST_TIMEOUT=300
PYTHON_CONNECT_TIMEOUT=15
PYTHON_INTERNAL_API_KEY=change-me-shared-secret
QUEUE_CONNECTION=redis
REDIS_CLIENT=predis
REDIS_HOST=127.0.0.1
REDIS_PORT=6379


### 2.3 Python

cd iust-rag-python   # یا Version_4
cp .env.example .env
# پر کردن secretها؛ LARAVEL__INTERNAL_API_KEY = همان کلید لاراول

python -m venv .venv
# Windows:
.\.venv\Scripts\activate
pip install -r requirements.txt

# PowerShell — ضروری مگر pip install -e .
$env:PYTHONPATH = "src"
uvicorn main:app --host 0.0.0.0 --port 8001 --reload --log-level info

ساختار کد: src/ (api, auth, config, core, ingest, models, utils, access_control)
ورود: main.py → api.app:create_app()
Production: DEBUG=false


### 2.4 Health
Python: GET http://127.0.0.1:8001/check یا /health
Laravel: بعد از login با Bearer


### 2.5 ترتیب smoke تست

redis-cli ping → PONG
Qdrant بالا
Python /check
POST /api/v1/user/login
آپلود سند → attach role/dept → publish
POST /api/v1/rag/ask
POST /api/v1/rag/ask-stream
Worker + POST /api/v1/rag/reembed + status

### ۳. قرارداد پاسخ
Laravel (ApiResponser):

{ "status": "success"|"error", "message": "...", "data": ... }
Python (ApiResponser):

{ "success": true|false, "message": "...", "data"|"errors": ... }

### ۴. API پایتون (:8001)

### 4.1 Health
|Method|Path|Body|
|------------|------------|------------|
|GET|/check|—|
|GET|/health|—|



### 4.2 Files
POST /api/v1/files/ingest

Content-Type: multipart/form-data
Auth: X-Internal-Key یا Bearer (نقش admin/developer)

|فیلد|اجباری|نوع| توضیح                                          |
|--------------------------|--------------------------|--------------------------|------------------------------------------------|
|file|بله|file| فایل سند                                       |
|department|خیر|string| پیش‌فرض public — پوشه زیر data/ |
|doc_uuid|خیر*|string| شناسه پایدار؛ برای overwrite توصیه می‌شود|
|roles|خیر|string (JSON array)| "مثال: [""public""]"|
|departments|خیر|string (JSON array)| "مثال: [""it""]"|
|permissions|خیر|string (JSON array)|
|status|خیر|string| draft / published/archived — پیش‌فرض published |
|version|خیر|int| پیش‌فرض 1|
|overwrite|خیر|string| 1/true یا 0/false — پیش‌فرض overwrite فعال     |


* اگر خالی باشد ممکن است uuid خودکار ساخته شود؛ برای sync با Laravel همان doc_uuid لاراول را بفرستید.
DELETE /api/v1/files/{doc_uuid}

Path: doc_uuid اجباری
Body: ندارد
Auth: Internal Key یا Bearer ادمین


## 4.3 Chat

### `POST /api/v1/chat/ask` — JSON

| فیلد           | اجباری   | نوع     | توضیح                                      |
|----------------|----------|---------|---------------------------------------------|
| `query`        | بله      | string  | ۱…۲۰۰۰ کاراکتر                             |
| `session_id`   | بله      | string  | شناسه نشست                                 |
| `selected_text`| خیر      | string  | متن کمکی (مثلاً از فایل)                   |
| `msg_id`       | خیر      | string  | رزرو؛ در Phase 1 معمولاً استفاده نمی‌شود  |
| `user_context` | شرطی*    | object  | اگر Bearer نباشد الزامی                    |

#### `user_context` (اگر ارسال شود)

| فیلد          | اجباری | نوع        |
|---------------|--------|------------|
| `user_id`     | بله    | int ≥ 1    |
| `username`    | بله    | string     |
| `roles`       | خیر    | string[]   |
| `departments` | خیر    | string[]   |
| `permissions` | خیر    | string[]   |

> * یا Bearer معتبر، یا `user_context` کامل.

---

### `POST /api/v1/chat/ask/stream`

- همان Body JSON مثل `/ask`
- پاسخ: **SSE** (`meta`, `sources`, `token`, `done`، در خطا `error`)

---

### `POST /api/v1/chat/ask_with_file` — multipart

| فیلد           | اجباری  | نوع         | توضیح                                              |
|----------------|---------|-------------|-----------------------------------------------------|
| `query`        | بله     | form string |                                                     |
| `session_id`   | بله     | form string |                                                     |
| `file`         | بله     | file        |                                                     |
| `user_context` | شرطی    | form string | JSON رشته‌ای از object هویت؛ یا Bearer             |

---

## 4.4 Sync

**پیشوند:** `/api/v1/sync`  
**نیاز به کاربر/کلید با نقش:** `admin` | `developer` | `superadmin`

| Method   | Path                              | Body / Query                                      |
|----------|-----------------------------------|---------------------------------------------------|
| `GET`    | `/documents/{doc_uuid}`           | Query اختیاری: `limit` (۱–۱۰۰، پیش‌فرض ۲۰)       |
| `GET`    | `/documents/department/{department}` | Query: `limit` اختیاری                         |
| `DELETE` | `/documents/{doc_uuid}`           | —                                                 |
| `PATCH`  | `/documents/{doc_uuid}/metadata`  | JSON آزاد metadata (کلیدهای ACL/status و …)      |
| `POST`   | `/collection/wipe`                | `confirm: bool` اجباری = `true`                   |
| `POST`   | `/data/cleanup`                   | پایین                                             |

### `POST /api/v1/sync/data/cleanup`

| فیلد                  | اجباری                  | پیش‌فرض | توضیح                        |
|-----------------------|-------------------------|---------|------------------------------|
| `dry_run`             | خیر                     | `true`  | فقط گزارش یتیم‌ها            |
| `confirm`             | وقتی `dry_run=false` بله | `false` | باید `true` باشد             |
| `remove_empty_dirs`   | خیر                     | `true`  |                              |
| `cleanup_temp_ingest` | خیر                     | `true`  |                              |

---

## ۵. API لاراول (`:8000/api`)

**Header رایج:**  
`Authorization: Bearer {token}`

برای بسیاری از searchها در کد قدیمی:  
`Accept: application/json` و درخواست Ajax.

---

### 5.1 Auth

#### `POST /api/v1/user/register` — JSON

| فیلد       | اجباری          |
|------------|-----------------|
| `name`     | بله             |
| `surname`  | بله             |
| `password` | بله             |
| `email`    | بله (unique)    |
| `username` | بله (unique)    |
| `ncode`    | بله             |
| `bio`      | خیر             |
| `phone`    | خیر (در save استفاده می‌شود) |

#### `POST /api/v1/user/login` — JSON

| فیلد       | اجباری |
|------------|--------|
| `username` | بله    |
| `password` | بله    |

#### `POST /api/v1/user/logout`
- Body: ندارد (با Bearer)

#### `POST /api/v1/user/change_password/{user_id}` — JSON

| فیلد             | اجباری          |
|------------------|-----------------|
| `oldPassword`    | بله             |
| `newPassword`    | بله (min 8)     |
| `confirmPassword`| بله             |

#### `GET|POST /api/v1/auth/verify-token`
- Body: ندارد — فقط Bearer (برای Python verify)

---

### 5.2 Users

| Method  | Path                                      | Body                                      |
|---------|-------------------------------------------|-------------------------------------------|
| `GET`   | `/api/v1/users`                           | —                                         |
| `POST`  | `/api/v1/user/store`                      | فیلدهای کاربر (مطابق مدل؛ ادمین)         |
| `POST`  | `/api/v1/upload/profile_pic`              | multipart: فایل پروفایل (طبق کنترلر)     |
| `GET`   | `/api/v1/user/show/{user_id}`             | —                                         |
| `POST`  | `/api/v1/user/update/{user_id}`           | فیلدهای قابل ویرایش (اختیاری)            |
| `GET`   | `/api/v1/user/delete/{user_id}`           | —                                         |
| `GET`   | `/api/v1/user/restore/{user_id}`          | —                                         |
| `POST`  | `/api/v1/user/search`                     | همه اختیاری: `name`, `surname`, `username`, `email`, … |
| `GET`   | `/api/v1/user/remove_pic/{user_id}`       | —                                         |
| `POST`  | `/api/v1/user/change/pass/{user_id}`      | پسورد (طبق کنترلر User)                  |
| `GET`   | `/api/v1/user/check_permission/{user_id}/{permission}` | —                              |
| `GET`   | `/api/v1/user/attach_role/{user_id}/{role_id}` | —                                   |
| `GET`   | `/api/v1/user/detach_role/{user_id}/{role_id}` | —                                   |
| `POST`  | `/api/v1/user/sync_roles/{user_id}`       | آرایه role idها (طبق کنترلر Relation)    |
| `GET`   | `/api/v1/user/attach_department/{user_id}/{dept_id}` | —                             |
| `GET`   | `/api/v1/user/detach_department/{user_id}/{dept_id}` | —                             |
| `POST`  | `/api/v1/user/sync_departments/{user_id}` | آرایه dept idها                           |

> برای `store` / `update` / `sync_*` دقیق‌ترین منبع validation داخل کنترلر مربوط است؛ path-paramها جایگزین Body هستند مگر `sync`.

---

### 5.3 Roles / Permissions / Departments

الگو یکسان است:

| عمل              | Method | Body                                      |
|------------------|--------|-------------------------------------------|
| index            | `GET`  | —                                         |
| store            | `POST` | فیلدهای عنوان (مثلاً `title_en`, `title_fa` — طبق کنترلر) |
| show             | `GET .../{id}` | —                                   |
| update           | `POST .../update/{id}` | فیلدهای اختیاری عنوان         |
| delete / restore | `GET`  | —                                         |
| attach / detach  | `GET` با دو id در path | —                          |
| sync             | `POST` | لیست idها در Body                         |

**پیشوندها:**
- **Roles:** `/api/v1/roles`, `/api/v1/role/...`
- **Permissions:** `/api/v1/permissions`, `/api/v1/permission/...`
- **Departments:** `/api/v1/depts`, `/api/v1/dept/...`

`POST /api/v1/dept/search` — فیلترهای اختیاری عنوان.

---

### 5.4 Documents

#### `POST /api/v1/doc/upload` — multipart

| فیلد              | اجباری | توضیح                                      |
|-------------------|--------|---------------------------------------------|
| `file`            | بله    | فایل                                       |
| `file_name_show`  | بله    | عنوان نمایشی                               |
| `status`          | خیر    | `draft` \| `published` \| `archived` — پیش‌فرض `draft` |

#### `POST /api/v1/doc/update/{doc_id}` — JSON/form

| فیلد              | اجباری |
|-------------------|--------|
| `file_name`       | خیر    |
| `file_name_show`  | خیر    |

#### `POST /api/v1/doc/search`

همه اختیاری: `extension`, `file_name`, `file_name_show`, `doc_uuid`, `status`, `version`

**بدون Body (فقط path + Bearer)**

| Method  | Path                                      | توضیح                          |
|---------|-------------------------------------------|--------------------------------|
| `GET`   | `/api/v1/docs`                            | لیست                           |
| `GET`   | `/api/v1/doc/show/{doc_id}`               |                                |
| `GET`   | `/api/v1/doc/delete/{doc_id}`             | Qdrant → disk → MySQL          |
| `GET`   | `/api/v1/doc/get/{doc_id}`                | دانلود                         |
| `GET`   | `/api/v1/doc/get_base64/{doc_id}`         |                                |
| `GET`   | `/api/v1/doc/publish/{doc_id}`            | publish + ingest               |
| `GET`   | `/api/v1/doc/archive/{doc_id}`            | archive + حذف از Qdrant        |
| `GET`   | `/api/v1/doc/attach_role/{doc_id}/{role_id}` |                            |
| `GET`   | `/api/v1/doc/detach_role/{doc_id}/{role_id}` |                            |
| `POST`  | `/api/v1/doc/sync_roles/{doc_id}`         | Body: لیست role id             |
| `GET`   | `/api/v1/doc/attach_department/{doc_id}/{dept_id}` |                    |
| `GET`   | `/api/v1/doc/detach_department/{doc_id}/{dept_id}` |                    |
| `POST`  | `/api/v1/doc/sync_departments/{doc_id}`   | Body: لیست dept id             |
| `GET`   | `/api/v1/doc/attach_permission/{doc_id}/{permission_id}` |              |
| `GET`   | `/api/v1/doc/detach_permission/{doc_id}/{permission_id}` |              |
| `POST`  | `/api/v1/doc/sync_permissions/{doc_id}`   | Body: لیست permission id       |

---

### 5.5 Chat / Session / Message / Feedback

#### Session

| Method  | Path                                      | Body                                      |
|---------|-------------------------------------------|-------------------------------------------|
| `GET`   | `/api/v1/chat/sessions`                   | —                                         |
| `GET`   | `/api/v1/chat/user_sessions`              | —                                         |
| `POST`  | `/api/v1/chat/session/store`              | اختیاری: `title` (در صورت پشتیبانی کنترلر) |
| `GET`   | `/api/v1/chat/session/show/{session_id}`  | —                                         |
| `POST`  | `/api/v1/chat/session/update/{session_id}`| `title` اجباری                            |
| `GET`   | `/api/v1/chat/session/delete/{session_id}`| —                                         |
| `POST`  | `/api/v1/chat/session/search`             | اختیاری: `title`                          |

#### Message

| Method  | Path                                              | Body                                      |
|---------|---------------------------------------------------|-------------------------------------------|
| `GET`   | `/api/v1/chat/message/{session_id}`               | —                                         |
| `POST`  | `/api/v1/chat/message/store`                      | محتوا / role طبق کنترلر (`human`/`ai`)   |
| `GET`   | `/api/v1/chat/message/show/{msg_id}`              | —                                         |
| `POST`  | `/api/v1/chat/message/update/{msg_id}`            | فیلدهای محتوا اختیاری/الزامی طبق کنترلر  |
| `GET`   | `/api/v1/chat/message/delete/{msg_id}`            | —                                         |
| `GET`   | `/api/v1/chat/message/feedback/{msg_id}/{feedback}` | بدون Body — feedback در path           |
| `POST`  | `/api/v1/chat/message/search`                     | فیلترهای اختیاری                          |

#### Message files

| Method  | Path                                              | Body                          |
|---------|---------------------------------------------------|-------------------------------|
| `POST`  | `/api/v1/chat/message/upload/{message_id}`        | multipart: `file` اجباری     |
| `GET`   | `/api/v1/chat/message/file/get/{file_id}`         | —                             |
| `GET`   | `/api/v1/chat/message/file/get_base_64/{file_id}` | —                             |
| `POST`  | `/api/v1/chat/message/file/search`                | فیلتر اختیاری                 |

---

### 5.6 RAG (Gateway اصلی فرانت)

همه با `auth:sanctum`؛ askها با `throttle:rag`.

#### `POST /api/v1/rag/ask` — JSON

| فیلد                   | اجباری | توضیح                                              |
|------------------------|--------|-----------------------------------------------------|
| `query`                | بله    | string، ۱…۲۰۰۰                                     |
| `session_id`           | خیر    | int؛ اگر نباشد session جدید ساخته می‌شود           |
| `msg_id`               | خیر    | string                                              |
| `selected_text`        | خیر    | string                                              |
| `edit_of_message_id`   | خیر    | int — ویرایش شاخه پیام                             |
| `skip_cache`           | خیر    | bool — دور زدن Redis cache                         |

#### `POST /api/v1/rag/ask-stream` — JSON

| فیلد                   | اجباری |
|------------------------|--------|
| `query`                | بله    |
| `session_id`           | خیر    |
| `selected_text`        | خیر    |
| `edit_of_message_id`   | خیر    |

> (بدون `skip_cache` در validation فعلی)

#### `POST /api/v1/rag/ask-with-file` — multipart

| فیلد                   | اجباری                          |
|------------------------|---------------------------------|
| `query`                | بله                             |
| `file`                 | بله (max ~20MB در validation)   |
| `session_id`           | خیر                             |
| `edit_of_message_id`   | خیر                             |

#### مدیریت دانش / نگهداری

| Method  | Path                          | Body / Query                                      |
|---------|-------------------------------|---------------------------------------------------|
| `POST`  | `/api/v1/rag/cache/clear`     | معمولاً خالی                                      |
| `POST`  | `/api/v1/rag/reembed`         | خالی — صف job؛ پاسخ `job_key`                     |
| `GET`   | `/api/v1/rag/reembed/status`  | Query اجباری: `job_key`                           |
| `POST`  | `/api/v1/rag/wipe-collection` | در صورت نیاز `confirm: true` (طبق پیاده‌سازی کنترلر) |
| `POST`  | `/api/v1/rag/data-cleanup`    | هم‌تراز Python: `dry_run`, `confirm`, …           |

> نقش ادمین/developer برای `cache` / `reembed` / `wipe` / `cleanup`.

---

## ۶. جریان داده سند

```
upload (draft) → attach role/dept → publish → Python ingest → Qdrant
archive / delete → حذف از Qdrant (+ disk/MySQL در destroy)
MySQL + disk Laravel: منبع حقیقت متادیتا و فایل
Qdrant: فقط بردار/چانک
sources خالی در ask: پاسخ ثابت «اطلاعاتی پیدا نشد» بدون توهم LLM
```

### ۷. صف و Redis 

`php artisan queue:work redis --timeout=3600 --tries=1`

```
بدون worker، reembed در وضعیت queued می‌ماند.
Cache پاسخ: exact match روی query نرمال‌شده + ACL + generation؛ بعد از publish/destroy/invalidate پاک می‌شود.
```

### ۸. تعویض مدل Embedding

```
تغییر AI__EMBEDDING_* در .env پایتون
Restart Python
wipe-collection (با confirm)
reembed (+ worker)
در صورت نیاز data-cleanup
```