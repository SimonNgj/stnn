
// readCOMDlg.cpp : implementation file
//
// OpenCV 3.40
// VS 2015 debug

#include "stdafx.h"
#include "readCOM.h"
#include "readCOMDlg.h"
#include "afxdialogex.h"
#include <fstream>
#include <conio.h>
#include <windows.h>
#include <string>
#include <msclr\marshal_cppstd.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <cstring>

#using <System.dll>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace System;
using namespace System::IO::Ports;

std::ofstream myfile;
gcroot<SerialPort^> mySerialPort = nullptr;

int serialBUSY = 0;

cv::VideoCapture cap;
cv::VideoWriter video;
cv::Mat frame;

// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();
	

// Dialog Data
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()

// CreadCOMDlg dialog

CreadCOMDlg::CreadCOMDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CreadCOMDlg::IDD, pParent)
	, name(_T("COM2"))
	, filesave(_T("simon1"))
	, baud(115200)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CreadCOMDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT1, name);
	DDX_Text(pDX, IDC_EDIT3, filesave);
	DDX_Text(pDX, IDC_EDIT2, baud);
}

BEGIN_MESSAGE_MAP(CreadCOMDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CreadCOMDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CreadCOMDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDOK, &CreadCOMDlg::OnBnClickedOk)
ON_BN_CLICKED(IDC_BUTTON3, &CreadCOMDlg::OnBnClickedButton3)
ON_WM_TIMER()
END_MESSAGE_MAP()


// CreadCOMDlg message handlers

BOOL CreadCOMDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here
	CFont *m_Font1 = new CFont;
	m_Font1->CreatePointFont(200, "Arial Bold");
	CWnd* pWnd1 = GetDlgItem(IDC_EDIT1);
	if (pWnd1)
	{
		pWnd1->SetFont(m_Font1);
		m_Font1->Detach();
	}
	CFont *m_Font2 = new CFont;
	m_Font2->CreatePointFont(200, "Arial Bold");
	CWnd* pWnd2 = GetDlgItem(IDC_EDIT2);
	if (pWnd2)
	{
		pWnd2->SetFont(m_Font2);
		m_Font2->Detach();
	}

	CFont *m_Font3 = new CFont;
	m_Font3->CreatePointFont(350, "Arial Bold");
	CWnd* pWnd3 = GetDlgItem(IDC_EDIT3);
	if (pWnd3)
	{
		pWnd3->SetFont(m_Font3);
		m_Font3->Detach();
	}

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CreadCOMDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CreadCOMDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CreadCOMDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

//------------------------------- Sub-functions---------------------------------//
// Data receiver handler for serial port
static void DataReceivedHandler(Object^ sender, SerialDataReceivedEventArgs^ e)
{
	SerialPort^ sp = (SerialPort^)sender;
	String^ indata = sp->ReadExisting();

	std::string incomingData = msclr::interop::marshal_as<std::string>(indata);
	myfile << incomingData;
}

//Start serial port
void CreadCOMDlg::OnBnClickedButton1()
{
	UpdateData(true);

	//String^ comN = gcnew System::String(name);
	//mySerialPort = gcnew SerialPort(comN);
	//mySerialPort->BaudRate = baud;
	//mySerialPort->Parity = Parity::None;
	//mySerialPort->StopBits = StopBits::One;
	//mySerialPort->DataBits = 8;
	//mySerialPort->Handshake = Handshake::None;
	//mySerialPort->RtsEnable = true;

	//mySerialPort->DataReceived += gcnew SerialDataReceivedEventHandler(DataReceivedHandler);
	//mySerialPort->Open();

	//char *file_name = (char*)(LPCTSTR)filesave;
	//myfile.open(file_name);

	//UpdateData(false);

	//while (1)
	//{
	//	key = _getch();
	//	value = key;

	//	if (value == 97) // If press 'b', code will exit
	//	{
	//		String^ comN = gcnew System::String(name);
	//		mySerialPort = gcnew SerialPort(comN);
	//		mySerialPort->BaudRate = baud;
	//		mySerialPort->Parity = Parity::None;
	//		mySerialPort->StopBits = StopBits::One;
	//		mySerialPort->DataBits = 8;
	//		mySerialPort->Handshake = Handshake::None;
	//		mySerialPort->RtsEnable = true;

	//		mySerialPort->DataReceived += gcnew SerialDataReceivedEventHandler(DataReceivedHandler);
	//		mySerialPort->Open();

	//		char *file_name = (char*)(LPCTSTR)filesave;
	//		myfile.open(file_name);
	//	}
	//	else if (value == 99)
	//	{
	//		mySerialPort->Close();
	//		myfile.close();
	//	}
	//	else if (value == 98)
	//		break;
	//}
	//UpdateData(false);
}

// Stop serial port
void CreadCOMDlg::OnBnClickedButton2()
{
	mySerialPort->Close();
	myfile.close();
}

void CreadCOMDlg::OnBnClickedOk()
{
	CDialogEx::OnOK();
}

BOOL CreadCOMDlg::PreTranslateMessage(MSG* pMsg)
{
	UpdateData(true);

	if (pMsg->message == WM_KEYDOWN)
	{
		if ((pMsg->wParam == VK_F14) && (serialBUSY == 0))
		{
			String^ comN = gcnew System::String(name);
			mySerialPort = gcnew SerialPort(comN);
			mySerialPort->BaudRate = baud;
			mySerialPort->Parity = Parity::None;
			mySerialPort->StopBits = StopBits::One;
			mySerialPort->DataBits = 8;
			mySerialPort->Handshake = Handshake::None;
			mySerialPort->RtsEnable = true;

			mySerialPort->DataReceived += gcnew SerialDataReceivedEventHandler(DataReceivedHandler);
			mySerialPort->Open();
			Sleep(1);
			CString filesaveF = "./data/" + filesave + ".csv";
			char *file_name = (char*)(LPCTSTR)(filesaveF);
			myfile.open(file_name);
			serialBUSY = 1;

			cap.open(0);
			if (!cap.isOpened())  exit(1);
			int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
			int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
			cv::String tt = "./data/" + filesave + ".avi";
			video.open(tt, CV_FOURCC('M', 'J', 'P', 'G'), 10, cv::Size(frame_width, frame_height));
			SetTimer(1, 100, NULL);
		}
		else if ((pMsg->wParam == VK_F15) && (serialBUSY == 1))
		{
			Sleep(1);
			mySerialPort->Close();
			myfile.close();
			serialBUSY = 0;

			KillTimer(1);
			cap.release();
		}
	}
	return CDialog::PreTranslateMessage(pMsg);
	UpdateData(false);
}

void CreadCOMDlg::OnBnClickedButton3()
{
	//UpdateData(true);

	//cap.open(0);
	//if (!cap.isOpened()) exit(1);

	//int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	//int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	//cv::String tt = filesave + ".avi";
	//video.open(tt, CV_FOURCC('M', 'J', 'P', 'G'), 10, cv::Size(frame_width, frame_height));

	//while (1)
	//{
	//	cap >> frame;
	//	if (frame.empty()) break;

	//	//video.write(frame); 
	//	imshow("Video", frame);

	//	char c = (char)cv::waitKey(1);
	//	if (c == 27 || c == 126) break;
	//}

	//// When everything done, release the video capture and write object
	//cap.release();
	//video.release();

	//// Closes all the windows
	//cv::destroyAllWindows(); 

	//	UpdateData(false);
}

void CreadCOMDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: Add your message handler code here and/or call default
	bool bSuccess = cap.read(frame); // read a new frame from video

	//pyrDown(frame, frame);
	video.write(frame); 
	imshow("Videostream", frame);

	CDialogEx::OnTimer(nIDEvent);
}
