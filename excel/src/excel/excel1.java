package excel;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFCell;

public class excel1 {
	
	static XSSFRow row;
	static XSSFCell cell;
	static XSSFRow row1;
	static XSSFCell cell1;
	static XSSFRow row2;
	static XSSFCell cell2;
	static int count = 0;
	
	//**************words갯수, 출력갯수, 파일명3개바꾸고, 바꾸려는단어교체***************
	public static void main(String[] args) {		
		String words[] = new String[50];
		String sentences[] = new String[100];
		String save_sentences[] = new String[200];

		//단어 배열 만들기
		try {
			//파일 불러오기
			FileInputStream word = new FileInputStream("C:\\Users\\NA\\git\\Project\\Dataset\\데이터셋/2018_11_22_6차(강의이름).xlsx");
			XSSFWorkbook wordbook = new XSSFWorkbook(word);
			
			//sheet수 취득
			//int sheetCn = workbook.getNumberOfSheets();
			int sheetCn=1;			
			System.out.println("sheet수 : " + sheetCn);

			//시트만큼 반복
			for(int cn = 0; cn < sheetCn; cn++){
				System.out.println("취득하는 sheet 이름 : " + wordbook.getSheetName(cn));
				//0번째 sheet 정보 취득
				XSSFSheet sheet = wordbook.getSheetAt(cn);
				
				//취득된 sheet에서 rows수 취득
				int rows = sheet.getPhysicalNumberOfRows();
				System.out.println(wordbook.getSheetName(cn) + " sheet의 row수 : " + rows);

				//취득된 row에서 취득대상 cell수 취득
				int cells = sheet.getRow(cn).getPhysicalNumberOfCells();
				System.out.println(wordbook.getSheetName(cn) + " sheet의 row에 취득대상 cell수 : " + cells);

				for (int r = 0; r < rows; r++) {
					row = sheet.getRow(r); // row 가져오기
					if (row != null) {
						for (int c = 0; c < cells; c++) {
							cell = row.getCell(c);
							if (cell != null) {								
								words[r] = "" + cell.getStringCellValue();								
							} else {
								System.out.print("cell이 없습니다\t");
							}
						} 
					}
				} 
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		//문장 배열 만들기 + 저장
		try {
			FileInputStream sentence = new FileInputStream("C:\\Users\\NA\\git\\Project\\Dataset\\데이터셋/2018_12_02_8차(교수님강의2).xlsx");
			XSSFWorkbook sentencebook = new XSSFWorkbook(sentence);
			//sheet수 취득
			//int sheetCn = workbook.getNumberOfSheets();
			int sheetCn=1;
			System.out.println("sheet수 : " + sheetCn);

			//시트만큼 반복
			for(int cn = 0; cn < sheetCn; cn++){
				System.out.println("취득하는 sheet 이름 : " + sentencebook.getSheetName(cn));
				
				//0번째 sheet 정보 취득
				XSSFSheet sheet = sentencebook.getSheetAt(cn);
				
				//취득된 sheet에서 rows수 취득
				int rows = sheet.getPhysicalNumberOfRows();
				System.out.println(sentencebook.getSheetName(cn) + " sheet의 row수 : " + rows);

				//취득된 row에서 취득대상 cell수 취득
				int cells = sheet.getRow(cn).getPhysicalNumberOfCells();
				System.out.println(sentencebook.getSheetName(cn) + " sheet의 row에 취득대상 cell수 : " + cells);

				for (int r = 0; r < rows; r++) {
					row1 = sheet.getRow(r); // row 가져오기
					if (row1 != null) {
						for (int c = 0; c < cells; c++) {
							cell1 = row1.getCell(c);
							if (cell1 != null) {
								sentences[r] = "" + cell1.getStringCellValue();
								//랜덤으로 3개 문장 넣어서 저장
								for(int i=0;i<3;i++)
								{
									save_sentences[count] = sentences[r];
									count++;
								}
								count=count-3;
								for(int i=0;i<3;i++)
								{
									int random_num = (int)(Math.random()*words.length);
									save_sentences[count] = save_sentences[count].replace("[[강의명]]", words[random_num]);
									count++;
								}														
							} else {								
							}
						}						
					}
				} 
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		//엑셀로 내보내기
		XSSFWorkbook writebook = new XSSFWorkbook();
		XSSFSheet writesheet = writebook.createSheet("mySheet");
		
		//출력 row 생성
		//내보내기위해 저장먼저
		for(int i=0;i<save_sentences.length;i++) {
			row = writesheet.createRow(i);
			row.createCell(0).setCellValue(save_sentences[i]);
		}		
		/*
		//출력 row 생성
		row = sheet.createRow(1);
		//출력 cell 생성
		row.createCell(0).setCellValue("DATA 21");
		row.createCell(1).setCellValue("DATA 22");
		row.createCell(2).setCellValue("DATA 23");

		row = sheet.createRow(2);
		//출력 cell 생성
		row.createCell(0).setCellValue("DATA 31");
		row.createCell(1).setCellValue("DATA 32");
		row.createCell(2).setCellValue("DATA 33");
		*/

		// 출력 파일 위치및 파일명 설정
		FileOutputStream outFile;
		try {
			outFile = new FileOutputStream("강의교수님_181204.xlsx");
			writebook.write(outFile);
			outFile.close();
			System.out.println("파일생성 완료");

		} catch (Exception e) {
			e.printStackTrace();
		}		
	}
}