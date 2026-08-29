# ------------------------------ users ------------------------------ #
create table users(
	id int not null auto_increment primary key,
	`name` varchar(200),
  surname varchar(200),
  username varchar(200),
  email varchar(200),
  phone varchar(50),
  ncode varchar(50),
  `password` varchar(200),
  bio text,
  sso_user_id int,
  
	created_at datetime null DEFAULT CURRENT_TIMESTAMP,
	updated_at datetime null DEFAULT null,
	deleted_at datetime null DEFAULT null
);

# ------------------------------ roles ------------------------------ #
create table roles(
	id int not null auto_increment primary key,
	title_en varchar(200),
  title_fa varchar(200),
  
	created_at datetime null DEFAULT CURRENT_TIMESTAMP,
	updated_at datetime null DEFAULT null,
	deleted_at datetime null DEFAULT null
);

# ------------------------------ permissions ------------------------------ #
create table permissions(
	id int not null auto_increment primary key,
	title_en varchar(200),
  title_fa varchar(200),
  
	created_at datetime null DEFAULT CURRENT_TIMESTAMP,
	updated_at datetime null DEFAULT null,
	deleted_at datetime null DEFAULT null
);

# ------------------------------ departments ------------------------------ #
create table departments(
	id int not null auto_increment primary key,
	title_en varchar(200),
  title_fa varchar(200),
  dept_id int, 
  
  index depatrment_depatrment_index(dept_id),
	constraint depatrment_depatrment_fk foreign key (dept_id) references departments(id) on delete cascade on update cascade,
	
	created_at datetime null DEFAULT CURRENT_TIMESTAMP,
	updated_at datetime null DEFAULT null,
	deleted_at datetime null DEFAULT null
);

# ------------------------------ role-user ------------------------------ #
create table role_user(
	id int not null auto_increment primary key,
  role_id int,
  user_id int, 
  
  index role_user_role_index(role_id),
	constraint role_user_role_fk foreign key (role_id) references roles(id) on delete cascade on update cascade,
	
  index role_user_user_index(user_id),
	constraint role_user_user_fk foreign key (user_id) references users(id) on delete cascade on update cascade,
	
	created_at datetime null DEFAULT CURRENT_TIMESTAMP,
	updated_at datetime null DEFAULT null,
	deleted_at datetime null DEFAULT null
);

# ------------------------------ permission-role ------------------------------ #
create table permission_role(
	id int not null auto_increment primary key,
  role_id int,
  permission_id int, 
  
  index permission_role_role_index(role_id),
	constraint permission_role_role_fk foreign key (role_id) references roles(id) on delete cascade on update cascade,
	
  index permission_role_permission_index(permission_id),
	constraint permission_role_permission_fk foreign key (permission_id) references permissions(id) on delete cascade on update cascade,
	
	created_at datetime null DEFAULT CURRENT_TIMESTAMP,
	updated_at datetime null DEFAULT null,
	deleted_at datetime null DEFAULT null
);

# ------------------------------ department-user ------------------------------ #
create table department_user(
	id int not null auto_increment primary key,
	dept_id int, 
  user_id int, 

  index department_user_department_index(dept_id),
	constraint department_user_department_fk foreign key (dept_id) references departments(id) on delete cascade on update cascade,
	
  index department_user_user_index(user_id),
	constraint department_user_user_fk foreign key (user_id) references users(id) on delete cascade on update cascade,
	
	created_at datetime null DEFAULT CURRENT_TIMESTAMP,
	updated_at datetime null DEFAULT null,
	deleted_at datetime null DEFAULT null
);

# ------------------------------ documents ------------------------------ #
create table documents(
	id int not null auto_increment primary key,
	file_name varchar(200),
  file_name_show varchar(200),
  path text,
  extension varchar(200),
  doc_uuid varchar(36) unique not null,
  `status` enum('draft', 'published', 'archived') default 'draft',
  version int default 1,
  uploader_id int,
  
  index document_uploader_index(uploader_id),
	constraint document_uploader_fk foreign key (uploader_id) references users(id) on delete cascade on update cascade,
	
	created_at datetime null DEFAULT CURRENT_TIMESTAMP,
	updated_at datetime null DEFAULT null,
	deleted_at datetime null DEFAULT null
);

# ------------------------------ doc-role ------------------------------ #
create table doc_role(
	id int not null auto_increment primary key,
  role_id int,
  doc_id int, 
  
  index doc_role_role_index(role_id),
	constraint doc_role_role_fk foreign key (role_id) references roles(id) on delete cascade on update cascade,
	
  index doc_role_doc_index(doc_id),
	constraint doc_role_doc_fk foreign key (doc_id) references documents(id) on delete cascade on update cascade,
	
	created_at datetime null DEFAULT CURRENT_TIMESTAMP,
	updated_at datetime null DEFAULT null,
	deleted_at datetime null DEFAULT null
);

# ------------------------------ doc-permission ------------------------------ #
create table doc_permission(
	id int not null auto_increment primary key,
  doc_id int,
  permission_id int, 
  
  index doc_permission_doc_index(doc_id),
	constraint doc_permission_doc_fk foreign key (doc_id) references documents(id) on delete cascade on update cascade,
	
  index doc_permission_permission_index(permission_id),
	constraint doc_permission_permission_fk foreign key (permission_id) references permissions(id) on delete cascade on update cascade,
	
	created_at datetime null DEFAULT CURRENT_TIMESTAMP,
	updated_at datetime null DEFAULT null,
	deleted_at datetime null DEFAULT null
);

# ------------------------------ dept-doc ------------------------------ #
create table department_doc(
	id int not null auto_increment primary key,
	dept_id int, 
  doc_id int, 

  index department_doc_depatrment_index(dept_id),
	constraint department_doc_depatrment_fk foreign key (dept_id) references departments(id) on delete cascade on update cascade,
	
  index department_doc_doc_index(doc_id),
	constraint department_doc_doc_fk foreign key (doc_id) references documents(id) on delete cascade on update cascade,
	
	created_at datetime null DEFAULT CURRENT_TIMESTAMP,
	updated_at datetime null DEFAULT null,
	deleted_at datetime null DEFAULT null
);

# ------------------------------ chat_sessions ------------------------------ #
create table chat_sessions(
	id int not null auto_increment primary key,
	title varchar(200),
  user_id int, 
  
  index chat_session_user_index(user_id),
	constraint chat_session_user_fk foreign key (user_id) references users(id) on delete cascade on update cascade,
	
	created_at datetime null DEFAULT CURRENT_TIMESTAMP,
	updated_at datetime null DEFAULT null,
	deleted_at datetime null DEFAULT null
);

# ------------------------------ chat_messages ------------------------------ #
create table chat_messages(
	id int not null auto_increment primary key,
	content text,
  role enum('human', 'ai', 'system'),
  feedback enum('1', '0'),
  sources json,
  msg_id varchar(50) unique,
  session_id int, 
  
  index chat_message_session_index(session_id),
	constraint chat_message_session_fk foreign key (session_id) references chat_sessions(id) on delete cascade on update cascade,
	
	created_at datetime null DEFAULT CURRENT_TIMESTAMP,
	updated_at datetime null DEFAULT null,
	deleted_at datetime null DEFAULT null
);

# ------------------------------ chat_message_files ------------------------------ #
create table chat_message_files(
	id int not null auto_increment primary key,
	file_name varchar(200),
  path text,
  extension varchar(200),
  message_id int,
  
  index chat_message_file_message_index(message_id),
	constraint chat_message_file_message_fk foreign key (message_id) references chat_messages(id) on delete cascade on update cascade,
	
	created_at datetime null DEFAULT CURRENT_TIMESTAMP,
	updated_at datetime null DEFAULT null,
	deleted_at datetime null DEFAULT null
);

# ------------------------------ audit_logs ------------------------------ #
create table audit_logs(
	id int not null auto_increment primary key,
	action varchar(50),
  entity_type varchar(50),
  entity_id varchar(100),
  details json,
  ip_address varchar(50),
  user_id int,
  
  index audit_log_user_index(user_id),
	constraint audit_log_user_fk foreign key (user_id) references users(id) on delete cascade on update cascade,
	
	created_at datetime null DEFAULT CURRENT_TIMESTAMP,
	updated_at datetime null DEFAULT null,
	deleted_at datetime null DEFAULT null
);

# ------------------------------ system_settings ------------------------------ #
create table system_settings(
	id int not null auto_increment primary key,
	setting_key varchar(200),
  setting_value text,
  group_key enum('general', 'ai', 'rag', 'security') default 'general',
  
	created_at datetime null DEFAULT CURRENT_TIMESTAMP,
	updated_at datetime null DEFAULT null,
	deleted_at datetime null DEFAULT null
);